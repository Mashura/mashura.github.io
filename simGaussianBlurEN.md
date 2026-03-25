---
layout: default
title: Real-Time Approximate Gaussian Blur
---

# Real-Time Approximate Gaussian Blur

For Gaussian blur on images, when the kernel is large (e.g. σ=150, sampling width ~450 pixels), even with horizontal-vertical separation, the computation is still substantial. On an iPhone 15 Pro, standard separable Gaussian blur on a 1920×1440 image can take hundreds of milliseconds.

Here I present a pyramid-downsampling-based approximation. Its runtime is nearly independent of σ—about 2ms on the same device at the same resolution—and σ can be adjusted continuously in real time. The tradeoff is that downsampling loses high-frequency information, so the result is an approximation of standard Gaussian blur, not an exact reproduction.

## Prerequisites

You should first make sure you fully **understand** the following:

1. Stacking multiple Gaussian filters still yields a Gaussian, satisfying: $σ^2=σ_1^2+σ_2^2$.

2. Each 2× downsampling step amplifies the effective blur by a factor of $4^i$ relative to the original input.

3. Bilinear downsampling has an approximate variance of $σ^2 = 1/3$, denoted $V_{ds} = 1/3$.

4. Bilinear upsampling has an approximate variance of $σ^2 = 1/6$, denoted $V_{us} = 1/6$.

5. Each level's downsample and blur can be merged into a single Metal Compute Shader—during the blur, sampling from the previous level's texture via the `sample` method performs the downsample at no extra cost. Each 16×16 Thread Group shares a 22×22 Half4 Threadgroup Memory block (halo of 3). Note that under this scheme, the very first operation is effectively a downsample rather than a blur, which may in principle cause jitter-like artifacts.

6. For Metal Shaders primarily targeting iPhone, splitting a 7×7 kernel into two horizontal-vertical passes with two read-write round-trips is not worthwhile. The GPU's wavefront scheduling mechanism effectively hides texture fetch latency.

## Terminology

1. We define Level 0 as the original image, Level 1 as (width/2, height/2), and so on.

2. For any $σ_{var}^2$, we use $V_{var}$ as shorthand.

## Computation

Since we use a 7×7 small kernel, we need to keep each level's σ within 0.66–1.5. The lower bound of 0.66 will prove useful later.

Suppose we're at level K. The equivalent $σ^2$ contributed by this level is:

$$
F(k,base)=σ^2=4^k*(σ_{downsample}^2+σ_{upscale}^2+4*σ_{base}^2)
$$

The cumulative equivalent $σ^2$ through level K is:

$$
C^K=σ^2=∑_{k=0}^{K-1}4^k*(σ_{downsample}^2+σ_{upscale}^2+4*σ_{base}^2)
$$

Which gives:

$$
C^K=σ^2=(4^K-1)/3*(σ_{downsample}^2+σ_{upscale}^2+4*σ_{base}^2)
$$

Now, for any given $σ_{target}$, we proceed as follows:

First check its $V_{target}$—if less than 2.25, just operate at level 0 directly. Skip everything below.

If greater than 2.25, fix $σ_{base} = 1.2$ and start accumulating from level 1. Say we've finished through level K and are now considering level K+1:

$$
V_{remain}=V_{target}-C^K
$$

If $V_{remain}>F(K+1,1.5)$, level K+1 can't fully absorb the remainder either. Set K+1's $σ_k$ to 1.2 as well and proceed to K+2.

If $F(K+1,0.66)≤V_{remain}≤F(K+1,1.5)$, level K+1 can exactly handle all of $V_{remain}$. Solve $F(K+1,base)=V_{remain}$ for the base value—that's the σ for level K+1, and planning is complete.

If $V_{remain}≤F(K+1,0.66)$, the σ required at level K+1 would be too small, potentially causing too many sample points to land in low-weight edge regions. So we skip level K+1 and push $V_{remain}$ back to the previous K levels for absorption.

Naturally, the next question is: how to distribute this residual across the first K levels?

I'll describe this following my original line of thinking—it may feel a bit roundabout, but it shouldn't be hard to follow.

First, prove Supplementary Lemma 1:

> For any $V_{remain}$, if we're allowed to add one blur pass with σ ∈ [0.66, 1.5] at any of the first K levels, at least one solution exists.

The proof is straightforward. The current topmost level is K. Adding one blur at this level yields a maximum $V_σ = V_{1.5} = 2.25 * 4^k$. The maximum $V_{remain}$ left over from level K+1 is $V_{remainmax}=F(K+1,0.66)=2.2424*4^k$.

So the upper bound of $V_{remain}$ can always be absorbed by level K (this is precisely why 0.66 was chosen as the lower bound).

Also, the higher the level, the smaller the performance cost.

Of course, if $V_{remain}$ is very small, it might fall below level K's minimum blur threshold.

Generalizing: adding one blur pass at level i yields a maximum of $V_{max_i}=V_{1.5}=2.25*4^i$. Level i+1's minimum is $V_{min_{i+1}}=V_{0.66}=4*0.66^2*4^i=1.7424*4^i$. Since the former exceeds the latter, the coverage across levels is seamless.

This completes the proof of Lemma 1.

Note that we initially used $σ_{base}$ of 1.2, which means if the redistributed $V_{remain}$ at some level is small enough, we can absorb it by increasing $σ_{base}$—leveraging the additivity of Gaussian variances. Raising $σ_{base}$ from 1.2 to 1.5 provides an additional $V = 2.25-1.44=0.81$. If $V_{remain}$ is below this, we can eliminate it with zero extra performance cost. Only when it exceeds 0.81 do we need an additional blur pass.

Extending further: in principle, if we remove the 0.66 lower bound, $V_{remain}$ can be fully absorbed by level K alone (ignoring floating-point precision and sampling errors), as long as the change to K's $σ_{base}$ is small enough. This is fundamentally different from adding an extra blur pass—not just in performance, but in sampling and computational precision, greatly expanding usability.

In other words, I believe the best design for absorbing $V_{remain}$ should be:

1. Working backwards from level K, compute what $σ_{inc}$ corresponds to $V_{remain}$. If it's greater than 0.9 (i.e. $sqrt(0.81)$), add a blur pass at level K—there's no better solution within this framework.

2. If less than 0.9, compute the increment to $σ_{base}$: $σ_{vary}=sqrt(σ_{inc}^2+1.2^2)-1.2$. If $σ_{vary}$ is below a predefined precision threshold (e.g. 0.01), push the operation down to level K-1 and repeat. This bottoms out at level 1, where we force a merge (since level 0 currently has no substantive operations—its downsample is already merged into level 1's computation, and level 0's cost is prohibitively high).

## Measured Performance

Test device: iPhone 15 Pro. Image size: 1920×1440. Timing via GPU Capture.

| Method | σ | Time |
|--------|---|------|
| This method | any | ~2ms |
| Standard separable Gaussian | 150 | 200–400ms |

Runtime is nearly independent of σ, because regardless of σ, each level always uses a fixed 7×7 small kernel. The only thing that increases is the number of pyramid levels (each additional level processes only 1/4 the pixels of the previous one).

## Limitations

1. Downsampling inherently loses high-frequency information, so the approximation differs from standard Gaussian blur in detail. The blur magnitude is similar, but not pixel-identical.
2. The $V_{ds}$ and $V_{us}$ values are theoretical approximations based on continuous distributions. Empirical calibration from actual render results can improve accuracy.
3. For very small σ (< 1.5), a single 7×7 Gaussian on the original image suffices, and this method offers no speedup.

## Relation to Existing Methods

Pyramid-downsampling for blur is not a new idea. The closest existing work is Dual Kawase (ARM, 2015). The main differences:

- Dual Kawase controls blur magnitude through pyramid level count. σ jumps discretely, and changing levels means rebuilding the pipeline.
- This method analytically computes each level's σ via closed-form formulas, enabling continuous real-time σ adjustment by simply recomputing parameters.
- This method has a systematic residual backtracking mechanism to precisely match the target σ. Dual Kawase has no corresponding design.
- Dual Kawase controls blur intensity through iteration counts and offsets, making it difficult to map directly to a specific mathematical variance. By calculating the variance contribution of each level, this method can directly accept the standard Gaussian σ as an input parameter. Under the same σ, the macroscopic blur intensity of both methods is generally consistent.

<video width="50%" controls>
  <source src="demo.mp4" type="video/mp4">
</video>
