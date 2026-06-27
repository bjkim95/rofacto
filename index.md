---
layout: project_page
permalink: /

title: Robot-Factored World Models via Robot Rendering
tab_title: Rofacto
social_title: Robot-Factored World Models via Robot Rendering
social_description: Action-conditioned video world models that expose actions as rendered robot geometry — a deployment-realistic, embodiment-shared visual interface.
description: Robot-Factored World Models move action realization and robot rendering outside the world model, exposing actions as visible robot geometry (URDF mesh RGB + end-effector/scene depth) for action-conditioned video generation.

venue: Preprint, 2026
authors:
  - name: Byungjun Kim
    affiliation: [1]
    equal_contributor: true
    homepage: https://bjkim95.github.io
  - name: Taeksoo Kim
    affiliation: [1]
    equal_contributor: true
    homepage: https://taeksuu.github.io/
  - name: Hyunsoo Cha
    affiliation: [1]
  - name: Hanbyul Joo
    affiliation: [1, 2]
    homepage: https://jhugestar.github.io/
affiliations:
  - Seoul National University
  - RLWRLD
equal_note: "*Equal contribution"

paper: static/pdf/rofacto.pdf
# arxiv: https://arxiv.org/abs/XXXX.XXXXX   # TODO: add once posted
# code:  https://github.com/snuvclab/rofacto  # TODO: released later
---

<!-- TL;DR -->
<div class="tldr">
  <b>TL;DR:</b> We move two robot-specific factors — <b>action realization</b> and
  <b>robot rendering</b> — outside the world model, so the model sees an action only as
  <b>visible robot geometry</b> and learns how the scene responds to it.
</div>

<!-- Teaser -->
<div class="video-placeholder">
  <i class="fas fa-play-circle"></i>
  <span>Teaser video</span>
  <small>drop <code>static/videos/teaser.mp4</code> here</small>
</div>

<!-- Abstract -->
<div class="columns is-centered has-text-centered">
  <div class="column is-four-fifths">
    <h2>Abstract</h2>
    <div class="content has-text-justified">
Action-conditioned video world models predict future observations from an initial observation and an action signal. In robotics, actions influence future observations through two distinct processes: they are first realized into robot motion by the robot body and controller, and the scene subsequently responds through contact outcomes and object motion. Conditioning directly on action commands asks the world model to learn the realization process itself, while conditioning on logged future states leaks the interaction outcomes it is meant to predict. We propose <b>robot-factored world models</b>, which move two robot-specific factors outside the world model. <b>First, action realization:</b> each command is rolled through the robot's own controller and kinematics into a deployment-available <i>nominal trajectory</i>. This middle signal avoids both robot-specific action-realization learning and future-state leakage. <b>Second, robot rendering:</b> this nominal trajectory is rendered through the robot URDF, factoring the robot's geometry, kinematics, and appearance out of the model and into explicit rendered robot geometry. To resolve depth ambiguity in the rendered interface, we further pair end-effector depth with scene depth, providing geometric cues for contact and occlusion beyond image-plane overlap. Together, camera-aware static RGB/depth context and rendered robot geometry form a shared visual world-model interface that remains consistent across viewpoints and robot embodiments. On DROID and RoboCasa-GR1, the rendered interface outperforms vector-conditioned baselines. We further demonstrate that the same rendered interface supports unseen robot embodiments at inference, and, as a downstream application, generate robot-interaction videos from retargeted human demonstrations.
    </div>
  </div>
</div>

---

## Method Overview

<figure class="fig">
  <img src="static/image/overview.png" alt="Robot-factored visual world-model interface">
  <figcaption>
    <b>Visual world-model interface.</b> Static context carries scene and viewpoint;
    rendered nominal robot geometry carries action; the diffusion model predicts the scene response.
  </figcaption>
</figure>

An action-conditioned robot world model predicts a future video $$\mathbf{V}_{1:F}$$ from the current observation and a proposed action sequence $$\boldsymbol{a}_{1:F}$$. Instead of conditioning on raw actions, we factor out two robot-specific steps as fixed preprocessing and leave the world model with the shared problem of predicting *scene response* around rendered robot motion:

$$\boldsymbol{q}_{1:F} = \Phi_R(\boldsymbol{a}_{1:F};\boldsymbol{q}_0), \qquad (\mathbf{M}^{\mathrm{rgb}}_{1:F}, \mathbf{D}^{\mathrm{eef}}_{1:F}) = \Pi_R(\boldsymbol{q}_{1:F};\mathcal{C}_{1:F}).$$

The realization operator $$\Phi_R$$ maps actions into a **nominal trajectory** (robot-only motion before scene interaction), and the rendering operator $$\Pi_R$$ projects it into camera-aligned **robot mesh RGB and end-effector depth**. A camera-aware static stream supplies scene appearance and depth, so the model learns

$$p_\theta\!\left(\mathbf{V}_{1:F} \mid \mathbf{B}^{\mathrm{rgb}}_{1:F}, \mathbf{D}^{\mathrm{scene}}_{1:F}, \mathbf{M}^{\mathrm{rgb}}_{1:F}, \mathbf{D}^{\mathrm{eef}}_{1:F}, \mathcal{T}\right),$$

where the text prompt $$\mathcal{T}$$ carries scene context only and excludes the intended action or outcome.

### 💡 Key Insight #1 — Action realization via the nominal trajectory

> **The right action signal lives between the raw command and the logged state: the controller-realized *nominal trajectory* is available at deployment, yet it does not leak scene interaction.**

<figure class="fig">
  <img src="static/image/realization_gaps.png" alt="Action-to-state realization gaps">
  <figcaption>
    <b>Action-to-state realization gaps.</b> (a) Robot-specific controllers and hardware constraints
    create a gap between raw actions and nominal trajectories. (b) Scene interaction creates a gap between
    nominal trajectories and realized states. The nominal trajectory is the deployment-available middle signal.
  </figcaption>
</figure>

<details>
<summary><strong>Why not raw actions or logged states?</strong></summary>

The mismatch between action signals decomposes into two gaps. The **action-realization gap** is the difference between the raw action and the controller-realized nominal motion; the **nominal–realized gap** is the difference between nominal robot motion and the state actually observed in a contact-rich rollout.

Conditioning on **raw actions** forces the model to additionally learn the robot-specific realization process. Conditioning on **logged future states** is visually aligned with the target video, but those states already encode contact, compliance, latency, and closed-loop corrections — i.e., they leak the interaction outcome the world model is meant to predict. Our factorization assigns the first gap to the robot-specific realization process $$\Phi_R$$ and leaves the second gap, together with object motion and occlusion, to the world model.

</details>

### 💡 Key Insight #2 — Robot rendering with depth

> **Rendering the nominal trajectory as URDF robot geometry makes the action visible in image space; pairing end-effector depth with scene depth resolves contact and occlusion beyond 2D overlap.**

<details>
<summary><strong>What the rendered interface provides</strong></summary>

The renderer $$\Pi_R$$ turns a nominal trajectory into **URDF mesh RGB** in the target camera frame, preserving link geometry, wrist offsets, gripper shape, and end-effector structure. This makes the action visible to the video model in the same visual coordinates as the target video and factors the robot's geometry and appearance out of the learned model.

RGB mesh rendering localizes the robot in image space, but it cannot tell whether an end-effector is in front of, behind, or in contact with an object. We therefore add **end-effector depth** $$\mathbf{D}^{\mathrm{eef}}_{1:F}$$ alongside **static scene depth** $$\mathbf{D}^{\mathrm{scene}}_{1:F}$$. Together these depth signals disambiguate proximity, likely contact, and occlusion beyond apparent 2D overlap — the model treats image-plane overlap as contact far less often.

We instantiate the scene-response model with a latent video diffusion backbone, following a residual-dynamics (full-mask video inpainting) formulation: the static context serves as the conditioning input, and the model learns interaction-induced scene changes conditioned on the static context and rendered robot geometry.

</details>

---

## Results

<section class="section results-section">
<h3 class="title is-4 has-text-centered">DROID — external / fixed cameras</h3>
<div class="results-grid">
  <div class="result-card">
    <!-- <video controls muted loop playsinline preload="metadata"><source src="static/videos/droid/result_1.mp4" type="video/mp4"></video> -->
    <div class="video-placeholder"><i class="fas fa-robot"></i><small>static/videos/droid/result_1.mp4</small></div>
    <div class="caption">Static context · Rendered robot · Prediction · GT</div>
  </div>
  <div class="result-card">
    <div class="video-placeholder"><i class="fas fa-robot"></i><small>static/videos/droid/result_2.mp4</small></div>
    <div class="caption">Static context · Rendered robot · Prediction · GT</div>
  </div>
  <div class="result-card">
    <div class="video-placeholder"><i class="fas fa-robot"></i><small>static/videos/droid/result_3.mp4</small></div>
    <div class="caption">Static context · Rendered robot · Prediction · GT</div>
  </div>
  <div class="result-card">
    <div class="video-placeholder"><i class="fas fa-robot"></i><small>static/videos/droid/result_4.mp4</small></div>
    <div class="caption">Static context · Rendered robot · Prediction · GT</div>
  </div>
</div>

<h3 class="title is-4 has-text-centered">RoboCasa-GR1 — humanoid manipulation</h3>
<div class="results-grid">
  <div class="result-card">
    <div class="video-placeholder"><i class="fas fa-robot"></i><small>static/videos/robocasa/result_1.mp4</small></div>
    <div class="caption">Static context · Rendered robot · Prediction · GT</div>
  </div>
  <div class="result-card">
    <div class="video-placeholder"><i class="fas fa-robot"></i><small>static/videos/robocasa/result_2.mp4</small></div>
    <div class="caption">Static context · Rendered robot · Prediction · GT</div>
  </div>
</div>
</section>

The rendered interface gives the video model direct pixel-space evidence of robot motion, so predicted scene changes are better localized around the robot and the contact region than vector- or pose-conditioned baselines. Rendered robot geometry localizes robot-driven scene changes, while depth helps resolve contact-relevant proximity and occlusion.

---

## Prompt Following

<figure class="fig">
  <img src="static/image/prompt_following.png" alt="Prompt-following probe">
  <figcaption>
    <b>Prompt-following probe.</b> Holding the initial scene fixed and editing only the rendered nominal
    motion changes the predicted scene response — the model uses rendered robot geometry as the action signal.
  </figcaption>
</figure>

## Zero-Shot Embodiment Generalization

<figure class="fig">
  <img src="static/image/embodiment.png" alt="Zero-shot embodiment composition">
  <figcaption>
    <b>Zero-shot embodiment composition.</b> HRDexDB contains an unseen xArm 6–Inspire F1 pairing.
    Because the interface represents action as rendered URDF geometry, unseen robot geometry is consumed by
    the same visual conditioning path and still drives the predicted scene response.
  </figcaption>
</figure>

## Application: Human Demonstration → Robot Video

<figure class="fig">
  <img src="static/image/human2robot.png" alt="Human demonstration to robot video">
  <figcaption>
    <b>Human demonstration to robot video.</b> Human manipulation videos (DexYCB) are retargeted to a robot
    and rendered as the same mesh-and-depth interface, then converted into a robot-interaction rollout —
    the world model consumes rendered robot geometry regardless of the motion source.
  </figcaption>
</figure>

<div class="video-placeholder">
  <i class="fas fa-play-circle"></i>
  <span>Human-to-robot rollouts</span>
  <small>drop <code>static/videos/human2robot.mp4</code> here</small>
</div>

---

## BibTeX

```bibtex
@article{kim2026rofacto,
  title   = {Robot-Factored World Models via Robot Rendering},
  author  = {Kim, Byungjun and Kim, Taeksoo and Cha, Hyunsoo and Joo, Hanbyul},
  journal = {arXiv preprint},
  year    = {2026}
}
```
