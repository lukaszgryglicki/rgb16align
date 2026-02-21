# rgb16align

`rgb16align` combines three separate captures (intended as **R**, **G**, **B** channels) into one aligned **16‑bit RGB** output image (**PNG** or **TIFF**).

Typical use‑case: you shoot 3 exposures of the same scene with the same lens/camera position, each time with a different filter, and want to align them (translation + small rotation) into a single RGB image.

---

## What it does

1. **Reads 3 input images** (JPG/PNG/TIF; grayscale or RGB/RGBA; 8‑bit or 16‑bit).
2. **Extracts one channel per input**:
   - **Red input**: if color, uses **R** channel; if grayscale, uses grayscale.
   - **Green input**: if color, uses **G** channel; if grayscale, uses grayscale.
   - **Blue input**: if color, uses **B** channel; if grayscale, uses grayscale.

   Note: OpenCV loads color as **BGR/BGRA**, so internally this maps to:
   - R = channel index 2
   - G = channel index 1
   - B = channel index 0

3. **Converts to 16‑bit** per channel internally:
   - 8‑bit → 16‑bit using full‑range scaling (`v16 = v8 * 257`)
   - 16‑bit remains 16‑bit
4. **Aligns images** using ECC (multi‑scale):
   - Default model: **Euclidean** (translation + rotation)
   - Default: **Affine fallback** if Euclidean fails
   - Default: aligns using **gradient magnitude (“edges”)** for better robustness when filters change intensity/contrast
5. **Automatically selects the best reference** (R or G or B) by choosing the one that yields the **largest overlap** after alignment.
6. **Warps** the other two channels into reference space.
7. **Crops** to the common valid overlap (unless disabled), then outputs:
   - **16‑bit RGB PNG** or **16‑bit RGB TIFF**

---

## Project layout

Create:

```text
rgb16align/
  Cargo.toml
  README.md
  src/
    main.rs
```

---

## Dependencies (Ubuntu)

You need:
- Rust toolchain (`cargo`, `rustc`)
- OpenCV dev libraries
- LLVM/Clang dev libraries (needed by `opencv` crate bindgen step)

Install packages:

```bash
sudo apt-get update
sudo apt-get install -y build-essential pkg-config clang \
  llvm-18-dev libclang-18-dev \
  libopencv-dev
```

If you previously saw `clang-sys` errors about `libclang.so` / `llvm-config`, the packages above are the fix.

---

## Build

From the project directory:

```bash
cargo build --release
```

Binary output:

```bash
./target/release/rgb16align
```

Optional: optimize for your CPU:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

---

## Run (sane defaults)

Normal usage requires only **3 inputs + 1 output**:

```bash
./target/release/rgb16align red_input.ext green_input.ext blue_input.ext out.png
```

Examples:

```bash
./target/release/rgb16align r.jpg g.jpg b.jpg out.png
./target/release/rgb16align r.tif g.tif b.tif out.tif
./target/release/rgb16align r.png g.png b.png out.tiff
```

Output extension must be one of: **.png**, **.tif**, **.tiff**

---

## Defaults (the “best/sane” mode)

By default, `rgb16align`:

- Uses `--reference auto` (tries R/G/B as reference, picks max overlap)
- Uses `--model euclidean`
- Uses `--affine-fallback=true`
- Uses edge‑based alignment (so `--no-edges` is NOT set)
- Uses alignment downscale limit: `--align-max-dim 3500`
- ECC pyramid base: `--pyramid-base-dim 800`
- ECC iterations per level: `--ecc-iters 300`
- ECC epsilon: `--ecc-eps 1e-6`
- Gaussian filter inside alignment: `--gauss 5`
- Crops to overlap with `--crop-margin 2`
- Reference selection overlap estimation at: `--ref-select-max-dim 1200`

---

## Options you may want to tweak

Show help:

```bash
./target/release/rgb16align --help
```

Force a specific reference (skips auto‑selection, faster):

```bash
./target/release/rgb16align r.tif g.tif b.tif out.png --reference green
```

Disable edge‑based alignment (use raw intensity instead):

```bash
./target/release/rgb16align r.tif g.tif b.tif out.png --no-edges
```

Use affine alignment always:

```bash
./target/release/rgb16align r.tif g.tif b.tif out.png --model affine
```

Disable affine fallback (Euclidean only):

```bash
./target/release/rgb16align r.tif g.tif b.tif out.png --affine-fallback=false
```

Disable cropping (keeps black borders from warping):

```bash
./target/release/rgb16align r.tif g.tif b.tif out.png --no-crop
```

Verbose (prints reference choice, ECC scores, warp matrices, crop rect):

```bash
./target/release/rgb16align r.tif g.tif b.tif out.png --verbose
```

---

## Notes / limitations

- This tool estimates **global motion** (translation + rotation; optional affine/homography).
- It does **not** correct:
  - parallax (scene depth + camera shift),
  - moving subjects between exposures,
  - lens distortion differences (usually minor if same lens/focus and small misalignment).
- Edge‑based ECC is usually more stable when filters change brightness/contrast strongly.

---

## Troubleshooting

### Build fails in `clang-sys` / `libclang.so` / `llvm-config`

Install:

```bash
sudo apt-get install -y llvm-18-dev libclang-18-dev
```

If needed, point Cargo at the exact tools/libs:

```bash
export LLVM_CONFIG_PATH="$(command -v llvm-config-18 || command -v llvm-config)"
export LIBCLANG_PATH="$(dirname "$(ldconfig -p | awk '/libclang\.so/{print $NF; exit}')")"
```

Then rebuild:

```bash
cargo clean
cargo build --release
```

### Runtime: “No overlap found”

Your frames may be too misaligned, or cropping/warping removed all valid intersection.

Try:
- `--model affine`
- increasing `--align-max-dim`
- `--no-crop` to inspect borders/valid area
