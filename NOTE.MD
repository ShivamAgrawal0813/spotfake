# Spot the Fake Photo — Note

## How I did it
When you take a photo of a screen, the screen itself leaves marks on the image — its pixel grid, its backlight, how it clips colours. So instead of training a CNN, I picked 7 simple features from each image and fed them into a logistic regression.

- **FFT peak energy** — a screen's pixel grid shows up as sharp spikes in the frequency domain that a real scene doesn't have (best signal, but gets weaker on 4K+ screens).
- **Laplacian variance** — measures sharpness/texture; screens often look too clean (OLED) or have a grid-like texture.
- **Saturation skew/p95** — screens clip colours to sRGB, so saturation bunches up near 1.0; real photos have a wider spread.
- **LBP entropy** — screen pixel grids repeat themselves, so the texture entropy is lower.
- **DCT block periodicity** — most screen photos are already JPEG, so photographing them again adds a second layer of 8×8 block compression.
- **Highlight ratio** — screens glare and reflect light, creating big flat overexposed patches.

I trained and tested this on 100 of my own phone photos (50 real, 50 screen, different lighting/angles/screens), using 5-fold cross-validation.

## Accuracy: **92.0% ± 6.8%** 
Most of the mistakes happen on screens photographed at a good angle in good light, where glare and moiré are weak — that's exactly where my strongest signal (FFT) struggles. Adding more of these "easy-looking" screen photos to training, plus a better moiré detector, should help close the gap.

## Latency **~113 ms per image** (112.76 ms median), tested on a laptop CPU.
Most of the time goes into the LBP step and the full-size FFT. Downsampling more before LBP and only using a small center crop for the FFT should bring this well under 50 ms.

## Cost per image
- **On-device**: free — runs locally on the phone during capture, no network call needed.
- **Cloud** (if it were ever needed): a cheap ~$0.05/hr CPU instance at ~110 ms/image works out   to roughly **$1.50 per million images** (~$0.0015 per 1,000).

## Trade-off
This sits on the small/fast/cheap end: free on-device, near-instant, and you can explain exactly why a photo got flagged. A CNN would probably be more accurate but higher latency and a bigger app size. A cloud LLM call would be the most flexible option but costs around $1 per 1,000 images and is too slow to feel instant. Since this needs to run quietly during capture, classical CV felt like the right choice.

## What I'd improve with more time
1. Collect more hard examples — flattering-angle screen shots, more screen/printout types, and cases someone might use to try to fool the model.
2. log scores in production, retrain every month on images that were wrong or disputed, and watch if scores start drifting toward 0.5 —  that usually means a new type of screen is slipping past. 
3. all 6 features already have OpenCV implementations, so this can ship through OpenCV's mobile bindings or get exported via CoreML/TFLite  — since there's no real "model," just a handful of weights, it should stay well under 1 MB. 
4. plot precision-recall on a held-out set and lean toward catching more cheaters — e.g. flag anything above 0.6 for review, auto-block above 0.85, and put everything in between through manual review.
