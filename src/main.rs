use anyhow::{bail, Context, Result};
use clap::{ArgAction, Parser, ValueEnum};
use opencv::{
    core::{self, Rect, Scalar, Size, TermCriteria},
    imgcodecs, imgproc,
    prelude::*,
    video,
};
use std::path::{Path, PathBuf};

#[derive(Copy, Clone, Debug, ValueEnum, PartialEq, Eq)]
enum Reference {
    /// Try R/G/B as reference and pick the one with the largest overlap (default)
    Auto,
    Red,
    Green,
    Blue,
}

#[derive(Copy, Clone, Debug, ValueEnum, PartialEq, Eq)]
enum MotionModel {
    Translation,
    Euclidean,
    Affine,
    Homography,
}

impl MotionModel {
    fn to_ocv(self) -> i32 {
        match self {
            MotionModel::Translation => video::MOTION_TRANSLATION,
            MotionModel::Euclidean => video::MOTION_EUCLIDEAN,
            MotionModel::Affine => video::MOTION_AFFINE,
            MotionModel::Homography => video::MOTION_HOMOGRAPHY,
        }
    }

    fn warp_kind(self) -> WarpKind {
        match self {
            MotionModel::Homography => WarpKind::Homography,
            _ => WarpKind::Affine,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum WarpKind {
    Affine,     // 2x3
    Homography, // 3x3
}

#[derive(Clone, Debug)]
struct Warp {
    kind: WarpKind,
    mat: Mat, // CV_32F
}

#[derive(Parser, Debug)]
#[command(
    name = "rgb16align",
    about = "Align 3 channel images (R/G/B) and output 16-bit RGB PNG/TIFF using ECC alignment."
)]
struct Args {
    /// Input image for RED channel (grayscale or RGB/RGBA; if color uses its R channel)
    red: PathBuf,

    /// Input image for GREEN channel (grayscale or RGB/RGBA; if color uses its G channel)
    green: PathBuf,

    /// Input image for BLUE channel (grayscale or RGB/RGBA; if color uses its B channel)
    blue: PathBuf,

    /// Output file path: .png, .tif, or .tiff (16-bit RGB)
    out: PathBuf,

    /// Reference selection: auto (default), red, green, blue
    #[arg(long, value_enum, default_value_t = Reference::Auto)]
    reference: Reference,

    /// ECC motion model to estimate (default: affine = translation + rotation + scale + shear)
    #[arg(long, value_enum, default_value_t = MotionModel::Affine)]
    model: MotionModel,

    /// If --model euclidean fails, retry that alignment with affine (default: true).
    /// This is a bool VALUE option, so you can do --affine-fallback=false
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    affine_fallback: bool,

    /// Use intensity images for alignment instead of gradient magnitude (edges).
    /// Edges are usually more robust when filters change brightness/contrast.
    #[arg(long)]
    no_edges: bool,

    /// Maximum dimension (pixels) used for alignment (downscales large images for speed).
    #[arg(long, default_value_t = 4500)]
    align_max_dim: i32,

    /// Smallest pyramid level max dimension for ECC (coarsest scale target).
    #[arg(long, default_value_t = 800)]
    pyramid_base_dim: i32,

    /// ECC iterations per pyramid level
    #[arg(long, default_value_t = 1000)]
    ecc_iters: i32,

    /// ECC epsilon (termination threshold)
    #[arg(long, default_value_t = 1e-6)]
    ecc_eps: f64,

    /// Gaussian filter size used inside ECC (odd number; 0 disables)
    #[arg(long, default_value_t = 3)]
    gauss: i32,

    /// Crop margin (pixels) after overlap detection (reduces edge interpolation artifacts)
    #[arg(long, default_value_t = 2)]
    crop_margin: i32,

    /// Do not crop to overlap; keep full canvas (will include black borders)
    #[arg(long)]
    no_crop: bool,

    /// For auto reference selection: compute overlap on a reduced mask of at most this max dimension.
    /// Higher = more accurate selection, slower.
    #[arg(long, default_value_t = 2000)]
    ref_select_max_dim: i32,

    /// Print warp matrices, ECC scores, and reference selection info
    #[arg(long)]
    verbose: bool,
}

#[derive(Copy, Clone, Debug)]
enum ChannelPick {
    Red,
    Green,
    Blue,
}

fn main() -> Result<()> {
    let args = Args::parse();

    ensure_supported_output(&args.out)?;

    // Load each file as single-channel 16-bit (CV_16U).
    let mut r16 = read_single_channel_16u(args.red.to_str().unwrap(), ChannelPick::Red)
        .context("Failed to load red image")?;
    let mut g16 = read_single_channel_16u(args.green.to_str().unwrap(), ChannelPick::Green)
        .context("Failed to load green image")?;
    let mut b16 = read_single_channel_16u(args.blue.to_str().unwrap(), ChannelPick::Blue)
        .context("Failed to load blue image")?;

    // Center-crop all to the same dimensions (min width/height).
    let target_w = r16.cols().min(g16.cols()).min(b16.cols());
    let target_h = r16.rows().min(g16.rows()).min(b16.rows());
    r16 = center_crop(&r16, target_w, target_h)?;
    g16 = center_crop(&g16, target_w, target_h)?;
    b16 = center_crop(&b16, target_w, target_h)?;

    if args.verbose {
        eprintln!("Working size (center-cropped): {}x{}", target_w, target_h);
    }

    // Choose reference + compute warps.
    let selection = match args.reference {
        Reference::Auto => choose_best_reference(&r16, &g16, &b16, &args)?,
        Reference::Red | Reference::Green | Reference::Blue => {
            compute_warps_for_reference(&r16, &g16, &b16, args.reference, &args)?
        }
    };

    if args.verbose {
        eprintln!(
            "Chosen reference: {:?} (estimated overlap area for selection: {})",
            selection.reference, selection.selection_area
        );
        for (name, w, score, used) in selection.warp_reports.iter() {
            eprintln!("Alignment {}: model={:?}, ECC score={:.6}", name, used, score);
            eprintln!("{}", mat_to_string(&w.mat)?);
        }
    }

    // Warp channels into reference frame at full resolution.
    let dst_size = Size::new(target_w, target_h);

    let aligned_r = match selection.reference {
        Reference::Red => r16.clone(),
        _ => warp_u16(&r16, dst_size, selection.warp_r.as_ref().unwrap())?,
    };
    let aligned_g = match selection.reference {
        Reference::Green => g16.clone(),
        _ => warp_u16(&g16, dst_size, selection.warp_g.as_ref().unwrap())?,
    };
    let aligned_b = match selection.reference {
        Reference::Blue => b16.clone(),
        _ => warp_u16(&b16, dst_size, selection.warp_b.as_ref().unwrap())?,
    };

    // Compute final overlap crop rectangle (full-res masks) unless --no-crop.
    let crop_rect = if args.no_crop {
        Rect::new(0, 0, target_w, target_h)
    } else {
        overlap_rect_full(
            target_w,
            target_h,
            selection.warp_r.as_ref(),
            selection.warp_g.as_ref(),
            selection.warp_b.as_ref(),
            selection.reference,
            args.crop_margin,
        )?
    };

    if args.verbose {
        eprintln!(
            "Final crop rect: x={} y={} w={} h={}",
            crop_rect.x, crop_rect.y, crop_rect.width, crop_rect.height
        );
    }

    // Crop aligned channels.
    let r_roi = crop_mat(&aligned_r, crop_rect)?;
    let g_roi = crop_mat(&aligned_g, crop_rect)?;
    let b_roi = crop_mat(&aligned_b, crop_rect)?;

    // Merge as BGR (OpenCV convention) then write. Output file will be standard RGB.
    let mut chans = core::Vector::<Mat>::new();
    chans.push(b_roi);
    chans.push(g_roi);
    chans.push(r_roi);

    let mut bgr = Mat::default();
    core::merge(&chans, &mut bgr)?;

    // Write output (16-bit RGB PNG/TIFF via OpenCV codecs)
    let params = core::Vector::<i32>::new();
    imgcodecs::imwrite(args.out.to_str().unwrap(), &bgr, &params)
        .with_context(|| format!("Failed to write output: {:?}", args.out))?;

    Ok(())
}

fn ensure_supported_output(out: &Path) -> Result<()> {
    let ext = out
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match ext.as_str() {
        "png" | "tif" | "tiff" => Ok(()),
        _ => bail!(
            "Unsupported output extension '.{}' (use .png, .tif, or .tiff)",
            ext
        ),
    }
}

struct Selection {
    reference: Reference, // Red/Green/Blue (Auto resolved)
    warp_r: Option<Warp>,
    warp_g: Option<Warp>,
    warp_b: Option<Warp>,
    selection_area: i64, // overlap area estimate used for choosing reference
    warp_reports: Vec<(String, Warp, f64, MotionModel)>, // (label, warp, score, used_model)
}

fn choose_best_reference(r16: &Mat, g16: &Mat, b16: &Mat, args: &Args) -> Result<Selection> {
    let candidates = [Reference::Red, Reference::Green, Reference::Blue];

    let mut best: Option<Selection> = None;

    for &cand in &candidates {
        match compute_warps_for_reference(r16, g16, b16, cand, args) {
            Ok(sel) => {
                if args.verbose {
                    eprintln!(
                        "Candidate {:?}: overlap area estimate = {}",
                        cand, sel.selection_area
                    );
                }
                let take = match &best {
                    None => true,
                    Some(b) => {
                        if sel.selection_area > b.selection_area {
                            true
                        } else if sel.selection_area == b.selection_area {
                            // tie-break: higher minimum ECC score across its two alignments
                            min_ecc_score(&sel.warp_reports) > min_ecc_score(&b.warp_reports)
                        } else {
                            false
                        }
                    }
                };
                if take {
                    best = Some(sel);
                }
            }
            Err(e) => {
                if args.verbose {
                    eprintln!("Candidate {:?} failed: {:#}", cand, e);
                }
            }
        }
    }

    best.ok_or_else(|| anyhow::anyhow!("All reference candidates failed to align (R/G/B)."))
}

fn min_ecc_score(reports: &[(String, Warp, f64, MotionModel)]) -> f64 {
    reports
        .iter()
        .map(|(_, _, s, _)| *s)
        .fold(f64::INFINITY, |a, b| a.min(b))
}

fn compute_warps_for_reference(
    r16: &Mat,
    g16: &Mat,
    b16: &Mat,
    reference: Reference,
    args: &Args,
) -> Result<Selection> {
    if reference == Reference::Auto {
        bail!("Internal error: compute_warps_for_reference called with Auto");
    }

    let (ref_name, ref_img) = match reference {
        Reference::Red => ("R", r16),
        Reference::Green => ("G", g16),
        Reference::Blue => ("B", b16),
        Reference::Auto => unreachable!(),
    };

    if args.verbose {
        eprintln!("Evaluating reference {:?} ({})", reference, ref_name);
    }

    let mut warp_r: Option<Warp> = None;
    let mut warp_g: Option<Warp> = None;
    let mut warp_b: Option<Warp> = None;
    let mut warp_reports: Vec<(String, Warp, f64, MotionModel)> = Vec::new();

    // Align each non-reference channel into the reference frame.
    if reference != Reference::Red {
        let label = format!("R -> {}", ref_name);
        let (w, score, used_model) = estimate_warp_ecc_with_fallback(
            ref_img,
            r16,
            args.model,
            args.affine_fallback,
            args.align_max_dim,
            args.pyramid_base_dim,
            args.ecc_iters,
            args.ecc_eps,
            args.gauss,
            !args.no_edges,
            args.verbose,
            &label,
        )
        .context("Alignment failed for R")?;
        warp_r = Some(w.clone());
        warp_reports.push((label, w, score, used_model));
    }

    if reference != Reference::Green {
        let label = format!("G -> {}", ref_name);
        let (w, score, used_model) = estimate_warp_ecc_with_fallback(
            ref_img,
            g16,
            args.model,
            args.affine_fallback,
            args.align_max_dim,
            args.pyramid_base_dim,
            args.ecc_iters,
            args.ecc_eps,
            args.gauss,
            !args.no_edges,
            args.verbose,
            &label,
        )
        .context("Alignment failed for G")?;
        warp_g = Some(w.clone());
        warp_reports.push((label, w, score, used_model));
    }

    if reference != Reference::Blue {
        let label = format!("B -> {}", ref_name);
        let (w, score, used_model) = estimate_warp_ecc_with_fallback(
            ref_img,
            b16,
            args.model,
            args.affine_fallback,
            args.align_max_dim,
            args.pyramid_base_dim,
            args.ecc_iters,
            args.ecc_eps,
            args.gauss,
            !args.no_edges,
            args.verbose,
            &label,
        )
        .context("Alignment failed for B")?;
        warp_b = Some(w.clone());
        warp_reports.push((label, w, score, used_model));
    }

    // Compute overlap area estimate for reference selection (scaled mask for speed).
    let w = ref_img.cols();
    let h = ref_img.rows();
    let area = overlap_area_estimate(
        w,
        h,
        warp_r.as_ref(),
        warp_g.as_ref(),
        warp_b.as_ref(),
        reference,
        args.ref_select_max_dim,
    )?;

    Ok(Selection {
        reference,
        warp_r,
        warp_g,
        warp_b,
        selection_area: area,
        warp_reports,
    })
}

/// ECC with optional fallback:
/// - if requested_model == Euclidean and affine_fallback == true:
///     try Euclidean; if error (or non-finite score), try Affine.
/// - otherwise: just run requested_model.
fn estimate_warp_ecc_with_fallback(
    template16: &Mat,
    input16: &Mat,
    requested_model: MotionModel,
    affine_fallback: bool,
    align_max_dim: i32,
    pyramid_base_dim: i32,
    ecc_iters: i32,
    ecc_eps: f64,
    gauss_filt_size: i32,
    use_edges: bool,
    verbose: bool,
    label: &str,
) -> Result<(Warp, f64, MotionModel)> {
    if requested_model == MotionModel::Euclidean && affine_fallback {
        let euclid = estimate_warp_ecc(
            template16,
            input16,
            MotionModel::Euclidean,
            align_max_dim,
            pyramid_base_dim,
            ecc_iters,
            ecc_eps,
            gauss_filt_size,
            use_edges,
        );

        match euclid {
            Ok((w, score)) if score.is_finite() => return Ok((w, score, MotionModel::Euclidean)),
            Ok((_w, score)) => {
                if verbose {
                    eprintln!(
                        "{}: Euclidean returned non-finite score ({:?}); falling back to affine",
                        label, score
                    );
                }
            }
            Err(e) => {
                if verbose {
                    eprintln!("{}: Euclidean failed; falling back to affine: {:#}", label, e);
                }
            }
        }

        let (w, score) = estimate_warp_ecc(
            template16,
            input16,
            MotionModel::Affine,
            align_max_dim,
            pyramid_base_dim,
            ecc_iters,
            ecc_eps,
            gauss_filt_size,
            use_edges,
        )
        .context("Affine fallback also failed")?;

        return Ok((w, score, MotionModel::Affine));
    }

    let (w, score) = estimate_warp_ecc(
        template16,
        input16,
        requested_model,
        align_max_dim,
        pyramid_base_dim,
        ecc_iters,
        ecc_eps,
        gauss_filt_size,
        use_edges,
    )?;

    Ok((w, score, requested_model))
}

/// Estimate warp from `input` to `template` using ECC on downscaled images + pyramid.
/// Returns warp matrix in FULL resolution coordinates + ECC score.
/// The warp is applied with WARP_INVERSE_MAP.
fn estimate_warp_ecc(
    template16: &Mat,
    input16: &Mat,
    model: MotionModel,
    align_max_dim: i32,
    pyramid_base_dim: i32,
    ecc_iters: i32,
    ecc_eps: f64,
    gauss_filt_size: i32,
    use_edges: bool,
) -> Result<(Warp, f64)> {
    let full_w = template16.cols();
    let full_h = template16.rows();
    let max_dim = full_w.max(full_h);

    // Downscale for alignment speed.
    let (templ_small, input_small, sx, sy) = if align_max_dim > 0 && max_dim > align_max_dim {
        let scale = align_max_dim as f64 / max_dim as f64;
        let w2 = ((full_w as f64) * scale).round().max(1.0) as i32;
        let h2 = ((full_h as f64) * scale).round().max(1.0) as i32;

        let mut t2 = Mat::default();
        let mut i2 = Mat::default();
        imgproc::resize(
            template16,
            &mut t2,
            Size::new(w2, h2),
            0.0,
            0.0,
            imgproc::INTER_AREA,
        )?;
        imgproc::resize(
            input16,
            &mut i2,
            Size::new(w2, h2),
            0.0,
            0.0,
            imgproc::INTER_AREA,
        )?;

        (t2, i2, w2 as f64 / full_w as f64, h2 as f64 / full_h as f64)
    } else {
        (template16.clone(), input16.clone(), 1.0, 1.0)
    };

    // Build alignment images.
    let templ_aln = make_alignment_image(&templ_small, use_edges, gauss_filt_size)?;
    let input_aln = make_alignment_image(&input_small, use_edges, gauss_filt_size)?;

    // Pyramid levels based on alignment image size.
    let mut levels: i32 = 0;
    let mut cur_max = templ_aln.cols().max(templ_aln.rows());
    while pyramid_base_dim > 0 && cur_max > pyramid_base_dim {
        levels += 1;
        cur_max /= 2;
        if levels > 8 {
            break;
        }
    }

    // Initialize warp matrix.
    let kind = model.warp_kind();
    let mut warp = match kind {
        WarpKind::Homography => {
            let mut w = Mat::new_rows_cols_with_default(3, 3, core::CV_32F, Scalar::all(0.0))?;
            *w.at_2d_mut::<f32>(0, 0)? = 1.0;
            *w.at_2d_mut::<f32>(1, 1)? = 1.0;
            *w.at_2d_mut::<f32>(2, 2)? = 1.0;
            w
        }
        WarpKind::Affine => {
            let mut w = Mat::new_rows_cols_with_default(2, 3, core::CV_32F, Scalar::all(0.0))?;
            *w.at_2d_mut::<f32>(0, 0)? = 1.0;
            *w.at_2d_mut::<f32>(1, 1)? = 1.0;
            w
        }
    };

    let mut gauss = gauss_filt_size;
    if gauss < 0 {
        gauss = 0;
    }
    if gauss > 0 && gauss % 2 == 0 {
        gauss += 1;
    }

    let criteria = TermCriteria::new(
        (core::TermCriteria_Type::COUNT as i32) | (core::TermCriteria_Type::EPS as i32),
        ecc_iters,
        ecc_eps,
    )?;

    let motion = model.to_ocv();
    let mut last_score = 0.0;

    // ECC coarse-to-fine.
    for lvl in (0..=levels).rev() {
        // Translation components scale ~2x between pyramid levels.
        if lvl != levels {
            upscale_warp_inplace(&mut warp, 2.0, 2.0, kind)?;
        }

        let scale = 1.0 / (1 << lvl) as f64;
        let w = ((templ_aln.cols() as f64) * scale).round().max(1.0) as i32;
        let h = ((templ_aln.rows() as f64) * scale).round().max(1.0) as i32;
        let sz = Size::new(w, h);

        let mut t_lvl = Mat::default();
        let mut i_lvl = Mat::default();
        imgproc::resize(&templ_aln, &mut t_lvl, sz, 0.0, 0.0, imgproc::INTER_AREA)?;
        imgproc::resize(&input_aln, &mut i_lvl, sz, 0.0, 0.0, imgproc::INTER_AREA)?;

        last_score = video::find_transform_ecc(
            &t_lvl,
            &i_lvl,
            &mut warp,
            motion,
            criteria,
            &Mat::default(), // no mask (stable default)
            gauss,
        )?;
    }

    // Convert warp from downscaled alignment coords back to full coords.
    let warp_full = scale_warp_to_full(&warp, sx, sy, kind)?;

    Ok((
        Warp {
            kind,
            mat: warp_full,
        },
        last_score,
    ))
}

/// Prepare an alignment image for ECC:
/// - convert to float [0..1]
/// - optionally compute gradient magnitude (edges)
fn make_alignment_image(src16: &Mat, use_edges: bool, gauss: i32) -> Result<Mat> {
    // float [0..1]
    let mut f = Mat::default();
    src16.convert_to(&mut f, core::CV_32F, 1.0 / 65535.0, 0.0)?;

    if gauss > 0 {
        let k = if gauss % 2 == 1 { gauss } else { gauss + 1 };
        let mut blurred = Mat::default();
        imgproc::gaussian_blur(
            &f,
            &mut blurred,
            Size::new(k, k),
            0.0,
            0.0,
            core::BORDER_DEFAULT,
        )?;
        f = blurred;
    }

    if !use_edges {
        return Ok(f);
    }

    // Sobel gradients
    let mut gx = Mat::default();
    let mut gy = Mat::default();
    imgproc::sobel(
        &f,
        &mut gx,
        core::CV_32F,
        1,
        0,
        3,
        1.0,
        0.0,
        core::BORDER_DEFAULT,
    )?;
    imgproc::sobel(
        &f,
        &mut gy,
        core::CV_32F,
        0,
        1,
        3,
        1.0,
        0.0,
        core::BORDER_DEFAULT,
    )?;

    let mut mag = Mat::default();
    core::magnitude(&gx, &gy, &mut mag)?;

    // Normalize for stability
    let mut mag_n = Mat::default();
    core::normalize(
        &mag,
        &mut mag_n,
        0.0,
        1.0,
        core::NORM_MINMAX,
        -1,
        &Mat::default(),
    )?;

    Ok(mag_n)
}

fn upscale_warp_inplace(warp: &mut Mat, fx: f64, fy: f64, kind: WarpKind) -> Result<()> {
    match kind {
        WarpKind::Homography => {
            // H = S * H * S^-1, where S = diag(fx, fy, 1)
            let s = [
                [fx as f32, 0.0, 0.0],
                [0.0, fy as f32, 0.0],
                [0.0, 0.0, 1.0],
            ];
            let si = [
                [(1.0 / fx) as f32, 0.0, 0.0],
                [0.0, (1.0 / fy) as f32, 0.0],
                [0.0, 0.0, 1.0],
            ];
            let h = mat3x3_to_array(warp)?;
            let tmp = mul3x3(&s, &h);
            let out = mul3x3(&tmp, &si);
            array_to_mat3x3_inplace(warp, &out)?;
        }
        WarpKind::Affine => {
            *warp.at_2d_mut::<f32>(0, 2)? *= fx as f32;
            *warp.at_2d_mut::<f32>(1, 2)? *= fy as f32;
        }
    }
    Ok(())
}

/// Convert warp from downscaled (sx,sy) back to full resolution.
fn scale_warp_to_full(warp_small: &Mat, sx: f64, sy: f64, kind: WarpKind) -> Result<Mat> {
    if (sx - 1.0).abs() < 1e-12 && (sy - 1.0).abs() < 1e-12 {
        return Ok(warp_small.clone());
    }

    match kind {
        WarpKind::Homography => {
            // H_full = S^-1 * H_small * S, where S = diag(sx, sy, 1)
            let s = [
                [sx as f32, 0.0, 0.0],
                [0.0, sy as f32, 0.0],
                [0.0, 0.0, 1.0],
            ];
            let si = [
                [(1.0 / sx) as f32, 0.0, 0.0],
                [0.0, (1.0 / sy) as f32, 0.0],
                [0.0, 0.0, 1.0],
            ];
            let h = mat3x3_to_array(warp_small)?;
            let tmp = mul3x3(&si, &h);
            let out = mul3x3(&tmp, &s);

            let mut w =
                Mat::new_rows_cols_with_default(3, 3, core::CV_32F, Scalar::all(0.0))?;
            array_to_mat3x3_inplace(&mut w, &out)?;
            Ok(w)
        }
        WarpKind::Affine => {
            let mut w = warp_small.clone();
            *w.at_2d_mut::<f32>(0, 2)? /= sx as f32;
            *w.at_2d_mut::<f32>(1, 2)? /= sy as f32;
            Ok(w)
        }
    }
}

fn overlap_area_estimate(
    w_full: i32,
    h_full: i32,
    warp_r: Option<&Warp>,
    warp_g: Option<&Warp>,
    warp_b: Option<&Warp>,
    reference: Reference,
    max_dim: i32,
) -> Result<i64> {
    // Build a smaller mask canvas for faster overlap evaluation
    let max_side = w_full.max(h_full);
    let (w2, h2, sx, sy) = if max_dim > 0 && max_side > max_dim {
        let scale = max_dim as f64 / max_side as f64;
        let w2 = ((w_full as f64) * scale).round().max(1.0) as i32;
        let h2 = ((h_full as f64) * scale).round().max(1.0) as i32;
        (w2, h2, w2 as f64 / w_full as f64, h2 as f64 / h_full as f64)
    } else {
        (w_full, h_full, 1.0, 1.0)
    };

    let size2 = Size::new(w2, h2);
    let base = Mat::new_rows_cols_with_default(h2, w2, core::CV_8U, Scalar::all(255.0))?;

    let mr = if reference == Reference::Red {
        base.clone()
    } else {
        let w = warp_r.context("Missing warp for R in overlap_area_estimate")?;
        let ws = scale_warp_to_scaled_space(w, sx, sy)?;
        warp_mask_u8(&base, size2, &ws)?
    };

    let mg = if reference == Reference::Green {
        base.clone()
    } else {
        let w = warp_g.context("Missing warp for G in overlap_area_estimate")?;
        let ws = scale_warp_to_scaled_space(w, sx, sy)?;
        warp_mask_u8(&base, size2, &ws)?
    };

    let mb = if reference == Reference::Blue {
        base.clone()
    } else {
        let w = warp_b.context("Missing warp for B in overlap_area_estimate")?;
        let ws = scale_warp_to_scaled_space(w, sx, sy)?;
        warp_mask_u8(&base, size2, &ws)?
    };

    let mut tmp = Mat::default();
    let mut inter = Mat::default();
    core::bitwise_and(&mr, &mg, &mut tmp, &Mat::default())?;
    core::bitwise_and(&tmp, &mb, &mut inter, &Mat::default())?;

    let rect = bounding_rect_nonzero_u8(&inter)?;
    Ok((rect.width as i64) * (rect.height as i64))
}

fn overlap_rect_full(
    w: i32,
    h: i32,
    warp_r: Option<&Warp>,
    warp_g: Option<&Warp>,
    warp_b: Option<&Warp>,
    reference: Reference,
    crop_margin: i32,
) -> Result<Rect> {
    let size = Size::new(w, h);
    let base = Mat::new_rows_cols_with_default(h, w, core::CV_8U, Scalar::all(255.0))?;

    let mr = if reference == Reference::Red {
        base.clone()
    } else {
        let w = warp_r.context("Missing warp for R in overlap_rect_full")?;
        warp_mask_u8(&base, size, w)?
    };

    let mg = if reference == Reference::Green {
        base.clone()
    } else {
        let w = warp_g.context("Missing warp for G in overlap_rect_full")?;
        warp_mask_u8(&base, size, w)?
    };

    let mb = if reference == Reference::Blue {
        base.clone()
    } else {
        let w = warp_b.context("Missing warp for B in overlap_rect_full")?;
        warp_mask_u8(&base, size, w)?
    };

    let mut tmp = Mat::default();
    let mut inter = Mat::default();
    core::bitwise_and(&mr, &mg, &mut tmp, &Mat::default())?;
    core::bitwise_and(&tmp, &mb, &mut inter, &Mat::default())?;

    let mut rect = bounding_rect_nonzero_u8(&inter)?;

    if crop_margin > 0 {
        let m = crop_margin;
        let nx = rect.x + m;
        let ny = rect.y + m;
        let nw = rect.width - 2 * m;
        let nh = rect.height - 2 * m;
        if nw <= 0 || nh <= 0 {
            bail!("Crop margin too large; overlap collapsed.");
        }
        rect = Rect::new(nx, ny, nw, nh);
    }

    Ok(rect)
}

fn scale_warp_to_scaled_space(w: &Warp, sx: f64, sy: f64) -> Result<Warp> {
    if (sx - 1.0).abs() < 1e-12 && (sy - 1.0).abs() < 1e-12 {
        return Ok(w.clone());
    }

    match w.kind {
        WarpKind::Affine => {
            if w.mat.rows() != 2 || w.mat.cols() != 3 {
                bail!("Affine warp must be 2x3");
            }
            let mut m = w.mat.clone();

            let a00 = *m.at_2d::<f32>(0, 0)?;
            let a01 = *m.at_2d::<f32>(0, 1)?;
            let a02 = *m.at_2d::<f32>(0, 2)?;
            let a10 = *m.at_2d::<f32>(1, 0)?;
            let a11 = *m.at_2d::<f32>(1, 1)?;
            let a12 = *m.at_2d::<f32>(1, 2)?;

            // For dst->src mapping, scaling both dst and src by S = diag(sx,sy):
            // A' = S*A*S^-1, t' = S*t
            let a01p = (sx / sy) as f32 * a01;
            let a10p = (sy / sx) as f32 * a10;
            let a02p = (sx as f32) * a02;
            let a12p = (sy as f32) * a12;

            *m.at_2d_mut::<f32>(0, 0)? = a00;
            *m.at_2d_mut::<f32>(0, 1)? = a01p;
            *m.at_2d_mut::<f32>(0, 2)? = a02p;
            *m.at_2d_mut::<f32>(1, 0)? = a10p;
            *m.at_2d_mut::<f32>(1, 1)? = a11;
            *m.at_2d_mut::<f32>(1, 2)? = a12p;

            Ok(Warp {
                kind: WarpKind::Affine,
                mat: m,
            })
        }
        WarpKind::Homography => {
            if w.mat.rows() != 3 || w.mat.cols() != 3 {
                bail!("Homography warp must be 3x3");
            }
            // H' = S * H * S^-1 where S = diag(sx, sy, 1)
            let s = [
                [sx as f32, 0.0, 0.0],
                [0.0, sy as f32, 0.0],
                [0.0, 0.0, 1.0],
            ];
            let si = [
                [(1.0 / sx) as f32, 0.0, 0.0],
                [0.0, (1.0 / sy) as f32, 0.0],
                [0.0, 0.0, 1.0],
            ];
            let h = mat3x3_to_array(&w.mat)?;
            let tmp = mul3x3(&s, &h);
            let out = mul3x3(&tmp, &si);

            let mut m =
                Mat::new_rows_cols_with_default(3, 3, core::CV_32F, Scalar::all(0.0))?;
            array_to_mat3x3_inplace(&mut m, &out)?;

            Ok(Warp {
                kind: WarpKind::Homography,
                mat: m,
            })
        }
    }
}

fn warp_u16(src: &Mat, dst_size: Size, warp: &Warp) -> Result<Mat> {
    let mut dst = Mat::default();
    match warp.kind {
        WarpKind::Homography => {
            imgproc::warp_perspective(
                src,
                &mut dst,
                &warp.mat,
                dst_size,
                imgproc::INTER_LINEAR + imgproc::WARP_INVERSE_MAP,
                core::BORDER_CONSTANT,
                Scalar::all(0.0),
            )?;
        }
        WarpKind::Affine => {
            imgproc::warp_affine(
                src,
                &mut dst,
                &warp.mat,
                dst_size,
                imgproc::INTER_LINEAR + imgproc::WARP_INVERSE_MAP,
                core::BORDER_CONSTANT,
                Scalar::all(0.0),
            )?;
        }
    }
    Ok(dst)
}

fn warp_mask_u8(mask: &Mat, dst_size: Size, warp: &Warp) -> Result<Mat> {
    let mut dst = Mat::default();
    match warp.kind {
        WarpKind::Homography => {
            imgproc::warp_perspective(
                mask,
                &mut dst,
                &warp.mat,
                dst_size,
                imgproc::INTER_NEAREST + imgproc::WARP_INVERSE_MAP,
                core::BORDER_CONSTANT,
                Scalar::all(0.0),
            )?;
        }
        WarpKind::Affine => {
            imgproc::warp_affine(
                mask,
                &mut dst,
                &warp.mat,
                dst_size,
                imgproc::INTER_NEAREST + imgproc::WARP_INVERSE_MAP,
                core::BORDER_CONSTANT,
                Scalar::all(0.0),
            )?;
        }
    }
    Ok(dst)
}

fn read_single_channel_16u(path: &str, pick: ChannelPick) -> Result<Mat> {
    let img = imgcodecs::imread(path, imgcodecs::IMREAD_UNCHANGED)
        .with_context(|| format!("imread failed for {path}"))?;
    if img.empty() {
        bail!("OpenCV returned empty image for {path}");
    }

    let channels = img.channels();
    let depth = img.depth();

    let mut single = Mat::default();
    if channels == 1 {
        single = img;
    } else if channels >= 3 {
        let idx = match pick {
            ChannelPick::Red => 2,   // R in BGR
            ChannelPick::Green => 1, // G
            ChannelPick::Blue => 0,  // B
        };
        core::extract_channel(&img, &mut single, idx)?;
    } else {
        // e.g. Gray+Alpha (2ch) - take channel 0
        core::extract_channel(&img, &mut single, 0)?;
    }

    // Normalize to CV_16U.
    let mut out16 = Mat::default();
    match depth {
        d if d == core::CV_16U => out16 = single,
        d if d == core::CV_8U => {
            // 8-bit -> 16-bit full-range scaling
            single.convert_to(&mut out16, core::CV_16U, 257.0, 0.0)?;
        }
        d if d == core::CV_32F || d == core::CV_64F => {
            // float handling: scale based on observed range.
            let mut f32m = Mat::default();
            single.convert_to(&mut f32m, core::CV_32F, 1.0, 0.0)?;

            let mut minv = 0.0;
            let mut maxv = 0.0;
            core::min_max_loc(
                &f32m,
                Some(&mut minv),
                Some(&mut maxv),
                None,
                None,
                &Mat::default(),
            )?;

            if (maxv - minv).abs() < 1e-12 {
                f32m.convert_to(&mut out16, core::CV_16U, 0.0, 0.0)?;
            } else {
                let (scale, shift) = if maxv <= 1.0 && minv >= 0.0 {
                    (65535.0, 0.0)
                } else if maxv <= 65535.0 && minv >= 0.0 {
                    (1.0, 0.0)
                } else {
                    let s = 65535.0 / (maxv - minv);
                    (s, -minv * s)
                };
                f32m.convert_to(&mut out16, core::CV_16U, scale, shift)?;
            }
        }
        _ => {
            // Fallback: convert without scaling.
            single.convert_to(&mut out16, core::CV_16U, 1.0, 0.0)?;
        }
    }

    Ok(out16)
}

fn center_crop(src: &Mat, w: i32, h: i32) -> Result<Mat> {
    let x = (src.cols() - w) / 2;
    let y = (src.rows() - h) / 2;
    let rect = Rect::new(x.max(0), y.max(0), w, h);

    let roi = Mat::roi(src, rect)?;
    Ok(roi.clone_pointee())
}

fn crop_mat(src: &Mat, rect: Rect) -> Result<Mat> {
    let roi = Mat::roi(src, rect)?;
    Ok(roi.clone_pointee())
}

/// Bounding rect of non-zero pixels in a CV_8U single-channel mask.
fn bounding_rect_nonzero_u8(mask: &Mat) -> Result<Rect> {
    if mask.typ() != core::CV_8U {
        bail!("Expected CV_8U mask");
    }
    let rows = mask.rows();
    let cols = mask.cols();

    let mut min_x = cols;
    let mut min_y = rows;
    let mut max_x = -1;
    let mut max_y = -1;

    if mask.is_continuous() {
        let data = mask.data_typed::<u8>()?;
        let stride = cols as usize;
        for y in 0..rows {
            let row = &data[(y as usize) * stride..(y as usize + 1) * stride];
            if !row.iter().any(|&v| v != 0) {
                continue;
            }
            for (x, &v) in row.iter().enumerate() {
                if v != 0 {
                    let xi = x as i32;
                    min_x = min_x.min(xi);
                    min_y = min_y.min(y);
                    max_x = max_x.max(xi);
                    max_y = max_y.max(y);
                }
            }
        }
    } else {
        for y in 0..rows {
            let row_ptr = mask.ptr(y)? as *const u8;
            for x in 0..cols {
                let v = unsafe { *row_ptr.add(x as usize) };
                if v != 0 {
                    min_x = min_x.min(x);
                    min_y = min_y.min(y);
                    max_x = max_x.max(x);
                    max_y = max_y.max(y);
                }
            }
        }
    }

    if max_x < 0 {
        bail!("No overlap found (mask is empty).");
    }

    Ok(Rect::new(min_x, min_y, max_x - min_x + 1, max_y - min_y + 1))
}

fn mat_to_string(m: &Mat) -> Result<String> {
    let r = m.rows();
    let c = m.cols();
    let mut s = String::new();
    s.push_str(&format!("warp {}x{}:\n", r, c));
    for y in 0..r {
        for x in 0..c {
            let v = *m.at_2d::<f32>(y, x)?;
            s.push_str(&format!("{:12.6} ", v));
        }
        s.push('\n');
    }
    Ok(s)
}

fn mat3x3_to_array(m: &Mat) -> Result<[[f32; 3]; 3]> {
    let mut a = [[0.0f32; 3]; 3];
    for y in 0..3usize {
        for x in 0..3usize {
            a[y][x] = *m.at_2d::<f32>(y as i32, x as i32)?;
        }
    }
    Ok(a)
}

fn array_to_mat3x3_inplace(m: &mut Mat, a: &[[f32; 3]; 3]) -> Result<()> {
    for y in 0..3usize {
        for x in 0..3usize {
            *m.at_2d_mut::<f32>(y as i32, x as i32)? = a[y][x];
        }
    }
    Ok(())
}

fn mul3x3(a: &[[f32; 3]; 3], b: &[[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut o = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            o[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    o
}
