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
    homepage: https://bjkim95.github.io
  - name: Taeksoo Kim
    affiliation: [1]
    homepage: https://taeksuu.github.io/
  - name: Hyunsoo Cha
    affiliation: [1]
    homepage: https://hyunsoocha.github.io/
  - name: Hanbyul Joo
    affiliation: [1, 2]
    homepage: https://jhugestar.github.io/
affiliations:
  - Seoul National University
  - RLWRLD

paper: static/pdf/rofacto.pdf
# arxiv: https://arxiv.org/abs/XXXX.XXXXX   # TODO: add once posted
code:  https://github.com/bjkim95/rofacto   # repo live for Watch/Star; code released later
---

<!-- TL;DR -->
<div class="tldr">
  <b>TL;DR:</b> We factor <b>action realization</b> and <b>robot appearance</b> out of the
  world model, presenting actions as <b>visible robot geometry</b> and leaving the model to
  learn how the scene responds.
</div>

<!-- Teaser -->
<video class="teaser-video" controls autoplay muted loop playsinline preload="metadata">
  <source src="static/videos/teaser.mp4" type="video/mp4">
</video>

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

## Results

<section class="section results-section">

<h3 class="title is-4 has-text-centered">DROID</h3>
<p class="results-sub">Fixed exterior camera · robot arm with parallel gripper</p>
<div class="carousel results-carousel">
  <div class="item">
    <div class="embod-example">
      <div class="strip-labels"><span>AdaLN<small>vector-based conditioning</small></span><span>Ours<small>rendering-based conditioning</small></span><span>GT</span></div>
      <video controls muted loop playsinline preload="metadata">
        <source src="static/videos/droid/rank041.mp4" type="video/mp4">
      </video>
    </div>
  </div>
  <div class="item">
    <div class="embod-example">
      <div class="strip-labels"><span>AdaLN<small>vector-based conditioning</small></span><span>Ours<small>rendering-based conditioning</small></span><span>GT</span></div>
      <video controls muted loop playsinline preload="metadata">
        <source src="static/videos/droid/rank005.mp4" type="video/mp4">
      </video>
    </div>
  </div>
  <div class="item">
    <div class="embod-example">
      <div class="strip-labels"><span>AdaLN<small>vector-based conditioning</small></span><span>Ours<small>rendering-based conditioning</small></span><span>GT</span></div>
      <video controls muted loop playsinline preload="metadata">
        <source src="static/videos/droid/rank014.mp4" type="video/mp4">
      </video>
    </div>
  </div>
  <div class="item">
    <div class="embod-example">
      <div class="strip-labels"><span>AdaLN<small>vector-based conditioning</small></span><span>Ours<small>rendering-based conditioning</small></span><span>GT</span></div>
      <video controls muted loop playsinline preload="metadata">
        <source src="static/videos/droid/rank108.mp4" type="video/mp4">
      </video>
    </div>
  </div>
</div>

<h3 class="title is-4 has-text-centered">RoboCasa-GR1</h3>
<p class="results-sub">Egocentric moving camera · humanoid with dexterous hands</p>
<div class="carousel results-carousel">
  <div class="item">
    <div class="embod-example">
      <div class="strip-labels"><span>AdaLN<small>vector-based conditioning</small></span><span>Ours<small>rendering-based conditioning</small></span><span>GT</span></div>
      <video controls muted loop playsinline preload="metadata">
        <source src="static/videos/robocasa/rank051.mp4" type="video/mp4">
      </video>
    </div>
  </div>
  <div class="item">
    <div class="embod-example">
      <div class="strip-labels"><span>AdaLN<small>vector-based conditioning</small></span><span>Ours<small>rendering-based conditioning</small></span><span>GT</span></div>
      <video controls muted loop playsinline preload="metadata">
        <source src="static/videos/robocasa/rank050.mp4" type="video/mp4">
      </video>
    </div>
  </div>
  <div class="item">
    <div class="embod-example">
      <div class="strip-labels"><span>AdaLN<small>vector-based conditioning</small></span><span>Ours<small>rendering-based conditioning</small></span><span>GT</span></div>
      <video controls muted loop playsinline preload="metadata">
        <source src="static/videos/robocasa/rank038.mp4" type="video/mp4">
      </video>
    </div>
  </div>
  <div class="item">
    <div class="embod-example">
      <div class="strip-labels"><span>AdaLN<small>vector-based conditioning</small></span><span>Ours<small>rendering-based conditioning</small></span><span>GT</span></div>
      <video controls muted loop playsinline preload="metadata">
        <source src="static/videos/robocasa/rank016.mp4" type="video/mp4">
      </video>
    </div>
  </div>
</div>

</section>

The rendered interface gives the video model direct pixel-space evidence of robot motion, so predicted scene changes are better localized around the robot and the contact region than vector- or pose-conditioned baselines. Rendered robot geometry localizes robot-driven scene changes, while depth helps resolve contact-relevant proximity and occlusion.

---

## Action Controllability

Holding the initial scene fixed and editing only the rendered nominal motion changes the predicted scene response — the model uses rendered robot geometry as the action signal.

<div class="carousel results-carousel">
  <div class="item">
    <div class="cf-fig">
      <div class="cf-collabels"><span>Original action</span><span>Edited action</span></div>
      <div class="cf-rowlabels"><span>Rendered action</span><span>Prediction</span></div>
      <video controls muted loop playsinline preload="metadata">
        <source src="static/videos/counterfactual/pick_tube.mp4" type="video/mp4">
      </video>
    </div>
    <p class="gap-cap">The edited trajectory is redirected toward a different object in the same scene.</p>
  </div>
  <div class="item">
    <div class="cf-fig">
      <div class="cf-collabels"><span>Original action</span><span>Edited action</span></div>
      <div class="cf-rowlabels"><span>Rendered action</span><span>Prediction</span></div>
      <video controls muted loop playsinline preload="metadata">
        <source src="static/videos/counterfactual/pick_tube_higher.mp4" type="video/mp4">
      </video>
    </div>
    <p class="gap-cap">The edited trajectory is re-solved to approach from higher above the object.</p>
  </div>
</div>

## Zero-Shot Embodiment Generalization

Since an action enters only as **rendered robot geometry**, robots never seen during training — a new arm–hand pairing, or even multiple arms — pass through the exact same interface and still drive a plausible scene response. This is nontrivial for vector- or pose-conditioned models, where a new embodiment brings a different action space and a new action-to-motion mapping to learn. Each clip shows **Mesh rendering (input)  ·  Ours (generated)  ·  GT**.

<section class="embod">

  <p class="embod-note">Beyond the rendered mesh shown here, the model also conditions on scene depth and end-effector depth; both are omitted from these strips for space.</p>

  <!-- ===== Group 1: unseen arm + hand composition (HRDexDB) ===== -->
  <div class="embod-group-title">Unseen Composition — Robot Arm + Dexterous Hand</div>
  <p class="embod-group-sub">An xArm 6 arm paired with an Inspire F1 hand — a combination never seen together in training. Evaluated on <a href="https://snuvclab.github.io/HRDexDB/" target="_blank">HRDexDB</a>.</p>

  <div class="embod-example">
    <div class="strip-labels"><span>Mesh rendering</span><span>Ours</span><span>GT</span></div>
    <video controls autoplay muted loop playsinline preload="metadata"
           poster="static/videos/embodiment/hrdex_banana.jpg">
      <source src="static/videos/embodiment/hrdex_banana.mp4" type="video/mp4">
    </video>
  </div>

  <!-- ===== Group 2: multiple arms (DexMimicGen) ===== -->
  <div class="embod-group-title">Multiple Robot Arms — Dual Robot Arms</div>
  <p class="embod-group-sub">Two robot arms composed into a single scene — a robot count outside the single-arm training setup. Evaluated on <a href="https://dexmimicgen.github.io/" target="_blank">DexMimicGen</a>.</p>

  <div class="embod-example">
    <div class="strip-labels"><span>Mesh rendering</span><span>Ours</span><span>GT</span></div>
    <video controls autoplay muted loop playsinline preload="metadata"
           poster="static/videos/embodiment/dexmimicgen_dual.jpg">
      <source src="static/videos/embodiment/dexmimicgen_dual.mp4" type="video/mp4">
    </video>
  </div>

</section>

## Application: Human Demonstration → Robot Video

<section class="embod">
<div class="carousel results-carousel">
  <div class="item">
    <div class="embod-example">
      <div class="strip-labels is-2"><span>DexYCB human demonstration</span><span>Ours — generated robot rollout</span></div>
      <video controls muted loop playsinline preload="metadata">
        <source src="static/videos/human2robot/detergent.mp4" type="video/mp4">
      </video>
    </div>
  </div>
  <div class="item">
    <div class="embod-example">
      <div class="strip-labels is-2"><span>DexYCB human demonstration</span><span>Ours — generated robot rollout</span></div>
      <video controls muted loop playsinline preload="metadata">
        <source src="static/videos/human2robot/wood_block.mp4" type="video/mp4">
      </video>
    </div>
  </div>
</div>
</section>

---

## Method Overview

<figure class="fig">
  <img src="static/image/overview.png" alt="Robot-factored visual world-model interface">
  <figcaption>
    <b>Visual world-model interface.</b> Static context carries scene and viewpoint;
    rendered nominal robot geometry carries action; the diffusion model predicts the scene response.
  </figcaption>
</figure>

Instead of conditioning the world model on raw action commands, we factor out two robot-specific steps as fixed preprocessing. First, each action is rolled through the robot's own controller and kinematics into a **nominal trajectory** — robot-only motion before any scene interaction. Second, this trajectory is rendered through the robot URDF into camera-aligned **robot mesh RGB and end-effector depth**. Paired with a camera-aware static stream (scene appearance and depth), these become the model's entire action interface, leaving it the single shared problem of predicting how the scene responds.

---

## Nominal Trajectory Conditioning

> **The right action signal lives between the raw command and the logged state: the controller-realized *nominal trajectory* is available at deployment and does not leak scene interaction.**

<div class="gap-block">
  <div class="gap-title">(a) Three action signals — which one should condition the world model?</div>
  <div class="gap-row is-3">
    <figure class="gap-cell">
      <video controls autoplay muted loop playsinline preload="metadata">
        <source src="static/videos/gaps/a_action_outline.mp4" type="video/mp4">
      </video>
      <figcaption>Raw action</figcaption>
    </figure>
    <figure class="gap-cell is-pivot">
      <video controls autoplay muted loop playsinline preload="metadata">
        <source src="static/videos/gaps/a_nominal_outline.mp4" type="video/mp4">
      </video>
      <figcaption>Nominal trajectory</figcaption>
    </figure>
    <figure class="gap-cell">
      <video controls autoplay muted loop playsinline preload="metadata">
        <source src="static/videos/gaps/a_logged_outline.mp4" type="video/mp4">
      </video>
      <figcaption>Realized state</figcaption>
    </figure>
  </div>
  <p class="gap-cap">DROID — the same episode rendered from each of the three signals.</p>
  <div class="gap-def">
    <b>Nominal trajectory</b> — the motion the robot is expected to follow under the commanded action
    <i>in the absence of any scene interaction</i>, obtained by rolling the command through the robot's
    own controller and kinematics.
  </div>
  <p>A <b>raw action</b> is a command, not a motion. It has not yet passed through the controller, so it reflects
  neither the controller's tracking behavior nor the robot's kinematic and actuation limits — and it therefore
  departs substantially from the state the robot actually reaches.</p>
  <p>The <b>realized state</b> is recorded in every dataset, but it is unavailable at inference: it is precisely the
  future the world model is asked to predict. Conditioning on it leaks the interaction outcome, since contact,
  compliance, and latency are already baked into the logged state.</p>
</div>

<div class="gap-block">
  <div class="gap-title">(b) The nominal–realized gap — what should the world model learn?</div>
  <div class="gap-row" style="--gap-ar: 416 / 206">
    <figure class="gap-cell is-pivot">
      <video controls autoplay muted loop playsinline preload="metadata">
        <source src="static/videos/gaps/b_nominal_outline.mp4" type="video/mp4">
      </video>
      <figcaption>Nominal trajectory</figcaption>
    </figure>
    <figure class="gap-cell">
      <video controls autoplay muted loop playsinline preload="metadata">
        <source src="static/videos/gaps/b_realized_outline.mp4" type="video/mp4">
      </video>
      <figcaption>Realized state</figcaption>
    </figure>
  </div>
  <p class="gap-cap">RoboCasa-GR1 — nominal trajectory and realized state on a contact-rich rollout.</p>
  <p>The two coincide while the robot moves freely, and separate once it meets the scene, where contact and
  compliance pull the realized motion away from the nominal plan. That divergence is not noise in the
  conditioning signal — it is the interaction itself, and predicting it is exactly what we leave to the
  world model.</p>
</div>

---

## Impact of Depth Conditioning

<section class="embod">

  <div class="embod-example">
    <div class="strip-labels"><span>Without depth</span><span>With depth</span><span>GT</span></div>
    <video controls autoplay muted loop playsinline preload="metadata">
      <source src="static/videos/depth/depth_c08_full.mp4" type="video/mp4">
    </video>
  </div>

  <div class="embod-example">
    <div class="strip-labels"><span>Without depth</span><span>With depth</span><span>GT</span></div>
    <video controls autoplay muted loop playsinline preload="metadata">
      <source src="static/videos/depth/depth_c02_full.mp4" type="video/mp4">
    </video>
  </div>

</section>

RGB mesh rendering places the robot only in the **image plane**, where overlap alone cannot tell a real touch from a robot simply passing in front of or behind an object. We pair **end-effector depth** with **scene depth** to make the model *depth-aware*, avoiding **false contact from image-plane overlap**.

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
