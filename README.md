# Rofacto — Project Page

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
  image/ figures cropped from the paper (overview, gaps, embodiment, ...)
  pdf/   rofacto.pdf  (linked by the "Paper" button)
  videos/  <-- drop result/teaser videos here (placeholders are in index.md)
```

## TODO before publishing

- [ ] Add `static/videos/teaser.mp4` and result clips (placeholders mark every slot).
- [ ] Fill in `arxiv:` and `code:` links in `index.md` front matter.
- [ ] Replace the `Preprint, 2026` venue once decided.
- [ ] Confirm author homepages / affiliations.

## Deploy (GitHub Pages)

This is the `gh-pages` branch. Enable Pages → *Deploy from branch* → `gh-pages` / root,
or push to a dedicated pages repo.
