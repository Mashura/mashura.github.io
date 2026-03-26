---
layout: default
title: Real-Time Approximate Gaussian Blur
---

# Real-Time Approximate Gaussian Blur

For Gaussian blur on images, when the kernel is very large (e.g. at σ=150, the sampling width is about 450 pixels), the computation is still heavy even with horizontal-vertical separable filtering. On an iPhone 15 Pro, performing a standard separable Gaussian blur on a 1920×1440 image can take hundreds of milliseconds.

Here we present an approximate scheme based on a downsampling pyramid. Its runtime is nearly independent of σ—on the same device and resolution it takes about 2ms, and σ can be adjusted continuously in real time. The trade-off is that downsampling loses high-frequency information, so the result is an approximation to the standard Gaussian blur, not an exact reproduction.

## Prerequisites

Make sure you fully **understand** the following prerequisites:

1. Cascading Gaussian filters is still a Gaussian filter, and the variances add: $σ^2=σ_1^2+σ_2^2$.

2. At the $i$-th level of downsampling, the blur variance scales by a factor of $4^i$ relative to the original input.

3. Using bilinear interpolation for downsampling, the variance is approximately $σ^2 = 1/3$, denoted $V_{ds} = 1/3$.

4. Using bilinear interpolation for upscaling, the variance is approximately $σ^2 = 1/6$, denoted $V_{us} = 1/6$.

5. Each level's downsample and blur can be merged into a single Metal Compute Shader—during the blur, the downsample is performed at no extra cost by sampling from the previous level's texture via the sample method. Each 16×16 thread group shares a 22×22 Half4 threadgroup memory block (with a halo of 3 on each side). Note, however, that under this scheme the first operation is effectively a downsample rather than a blur, which in principle may cause artifacts such as jittering.

6. For Metal Shaders, since the primary target device is iPhone, splitting a 7×7 kernel into two horizontal-vertical passes with two read-write rounds is not worthwhile for this kernel size. The GPU's wavefront scheduling mechanism can effectively hide texture fetch latency.

## Terminology

1. We define level 0 as the original image, level 1 as (width/2, height/2), and so on.

2. For any $σ_{var}^2$, we use $V_{var}$ to denote it.

## Computation

Since we use a small 7×7 kernel, we need to keep each level's σ within [0.66, 1.5]. The reason for choosing 0.66 as the lower bound will become clear later.

Suppose we are at level K. The equivalent $σ^2$ contributed by this level is:

$$
F(k,base)=σ^2=4^k*(σ_{downsample}^2+σ_{upscale}^2+4*σ_{base}^2)
$$

The cumulative equivalent $σ^2$ up through level K is:

$$
C^K=σ^2=∑_{k=0}^{K-1}4^k*(σ_{downsample}^2+σ_{upscale}^2+4*σ_{base}^2)
$$

Which simplifies to:

$$
C^K=σ^2=(4^K-1)/3*(σ_{downsample}^2+σ_{upscale}^2+4*σ_{base}^2)
$$

Now, for any given $σ_{target}$, we proceed as follows:

First check $V_{target}$. If it is less than 2.25, simply operate on level 0 directly. No further processing is needed.

If it is greater than 2.25, we fix $σ_{base} = 1.2$ and accumulate level by level starting from level 1. Suppose we have finished level K and are now evaluating level K+1:

$$
V_{remain}=V_{target}-C^K
$$

If $V_{remain}>F(K+1,1.5)$, level K+1 alone cannot absorb all of it. Set level K+1's $σ_k$ to 1.2 as well, and continue to level K+2.

If $F(K+1,0.66)≤V_{remain}≤F(K+1,1.5)$, level K+1 can exactly absorb all of $V_{remain}$. Solve $F(K+1,base)=V_{remain}$ for the base value—that gives level K+1's σ, and the plan is complete.

If $V_{remain}≤F(K+1,0.66)$, the σ required at level K+1 would be too small, potentially causing too many sample points to fall in the low-weight tail region. In this case we skip level K+1 and redistribute $V_{remain}$ back to the first K levels.

The question then becomes: how do we distribute this remainder across the first K levels?

I'll describe this following my original line of thinking—it may feel a bit roundabout, but it shouldn't be hard to follow.

First, prove Supplementary Lemma 1:

> For any $V_{remain}$, if we are allowed to add one blur pass with σ ∈ [0.66, 1.5] at any of the first K levels, at least one solution exists.

The proof is straightforward. The current topmost level is K. Adding one blur at this level yields a maximum $V_σ = V_{1.5} = 2.25 * 4^k$. The maximum $V_{remain}$ left over from level K+1 is $V_{remainmax}=F(K+1,0.66)=2.2424*4^k$.

So the upper bound of $V_{remain}$ can always be absorbed by level K (this is precisely why 0.66 was chosen as the lower bound).

Also, the higher the level, the smaller the performance cost.

Of course, if $V_{remain}$ is very small, it might fall below level K's minimum blur threshold.

Generalizing: adding one blur pass at level i yields a maximum of $V_{max_i}=V_{1.5}=2.25\*4^i$. Level i+1's minimum is $V_{min_i+1}=V_{0.66}=4\*0.66^2\*4^i=1.7424\*4^i$. Since the former exceeds the latter, the coverage across levels is seamless.

This completes the proof of Lemma 1.

Note that we initially used $σ_{base}$ of 1.2, which means if the redistributed $V_{remain}$ at some level is small enough, we can absorb it by increasing $σ_{base}$—leveraging the additivity of Gaussian variances. Raising $σ_{base}$ from 1.2 to 1.5 provides an additional $V = 2.25-1.44=0.81$. If $V_{remain}$ is below this, we can eliminate it with zero extra performance cost. Only when it exceeds 0.81 do we need an additional blur pass.

Extending further: in principle, if we remove the 0.66 lower bound, $V_{remain}$ can be fully absorbed by level K alone (ignoring floating-point precision and sampling errors), as long as the change to level K's $σ_{base}$ is small enough. This is fundamentally different from adding an extra blur pass—not just in performance, but in sampling and computational precision, greatly expanding usability.

In other words, I believe the best design for absorbing $V_{remain}$ should be:

1. Working backwards from level K, compute what $σ_{inc}$ corresponds to $V_{remain}$. If it is greater than 0.9 (i.e. $sqrt(0.81)$), add a blur pass at level K—there is no better solution within this framework.

2. If it is less than 0.9, compute the increment to $σ_{base}$: $σ_{vary}=sqrt(σ_{inc}^2+1.2^2)-1.2$. If $σ_{vary}$ is below a predefined precision threshold (e.g. 0.01), push the operation down to level K-1 and repeat. This bottoms out at level 1, where we force a merge (since under this scheme level 0 currently has no substantive operations—its downsample is already merged into level 1's computation, and level 0's cost is prohibitively high).

## Benchmark Data

Test device: iPhone 15 Pro. Image size: 1920×1440. Timing measured via GPU Capture.

| Method                    | σ   | Time        |
| ------------------------- | --- | ----------- |
| This method               | Any | ~2ms        |
| Standard separable Gauss  | 150 | 200–400ms   |

The runtime of this method is nearly independent of σ, because regardless of how large σ is, each level always uses a fixed 7×7 small kernel. The only thing that increases is the number of pyramid levels (each additional level processes only 1/4 the pixels of the previous one).

## Advantages

1. At the same σ, the result is visually close to standard Gaussian blur. Although not perfectly identical, you at least have a measurable reference. For non-strict comparisons it can directly substitute standard Gaussian blur (I think this is the most important advantage).
2. σ adjustment is real-time and continuous—only the rendering plan needs to be dynamically updated on the CPU side; no shader recompilation is needed on the GPU side.

## Limitations

1. Downsampling inherently loses high-frequency information, so the approximation differs from standard Gaussian blur in fine details. The blur magnitude is similar, but it is not pixel-accurate.
2. The values of $V_{ds}$ and $V_{us}$ are theoretical approximations based on continuous distributions. For higher-precision fitting, these two parameters can be calibrated against actual rendering results.
3. When σ is very small (< 1.5), a single 7×7 Gaussian pass on the original image suffices, and this method offers no speedup advantage.

<video width="50%" controls>
  <source src="demo.mp4" type="video/mp4">
</video>
