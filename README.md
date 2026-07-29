# RoFacto — Project Page

Project page for **Robot-Factored World Models via Robot Rendering**.

Built with [Jekyll](https://jekyllrb.com/) + [Bulma](https://bulma.io/), adapted from the
NeRFies / Jekyll academic project-page template.

## Local development

```bash
bundle install        # first time only
bundle exec jekyll serve
# open http://localhost:4000
```

If you have a global Jekyll install you can also just run `jekyll serve`.

## Structure

```
_config.yml                 # site config
_layouts/project_page.html  # page shell (header, links, footer, MathJax)
index.md                    # page content (front matter = title/authors/links)
static/
  css/   bulma.min.css, index.css
  js/    fontawesome, carousel/slider init
  image/ figures cropped from the paper (overview, gaps, embodiment, ...) + favicons
  videos/ teaser.mp4 plus result clips, grouped by experiment
          (droid/, robocasa/, depth/, gaps/, embodiment/, human2robot/, counterfactual/)
```

The paper is served from arXiv, not from this repo — the "arXiv" button reads the
`arxiv:` field in the `index.md` front matter.

## TODO

- [ ] Replace the `Preprint, 2026` venue once decided.

## Deploy (GitHub Pages)

This is the `gh-pages` branch. Enable Pages → *Deploy from branch* → `gh-pages` / root,
or push to a dedicated pages repo.
