// Video carousel autoplay when in view (desktop) or play/pause based on visibility (mobile)
function setupVideoCarouselAutoplay() {
    const carouselVideos = document.querySelectorAll('.results-carousel video');
    
    if (carouselVideos.length === 0) return;
    
    const isMobile = window.matchMedia("(max-width: 768px)").matches;
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            const video = entry.target;
            if (entry.isIntersecting && entry.intersectionRatio > 0.6) {
                // Video is sufficiently in view
                if (video.hasAttribute("autoplay") || !isMobile) {
                    // Desktop: autoplay if attribute exists
                    // Mobile: only play if user interacts (controls)
                    video.play().catch(e => {
                        // Autoplay failed, probably due to browser policy
                        console.log('Autoplay prevented:', e);
                    });
                }
            } else {
                // Video is out of view, pause it
                video.pause();
            }
        });
    }, {
        threshold: [0, 0.6, 1.0] // Multiple thresholds for better detection
    });
    
    carouselVideos.forEach(video => {
        observer.observe(video);
    });
}

$(document).ready(function() {

    // Check if mobile
    const isMobile = window.matchMedia("(max-width: 768px)").matches;
    var carousels = [];

    // Only initialize bulma-carousel on desktop
    if (!isMobile) {
      var options = {
        slidesToScroll: 1,
        slidesToShow: 1,
        loop: true,
        infinite: true,
        autoplay: false, // Slides advance on user navigation only
      }

      // Initialize all div with carousel class (desktop only)
      carousels = bulmaCarousel.attach('.carousel', options);

      // Start the clip on whichever slide is showing. Clips loop, so they never
      // fire `ended` and the carousel does not advance on its own; the
      // IntersectionObserver below pauses slides once they scroll out of view.
      carousels.forEach((carousel) => {
        const carouselElement = carousel.element;

        const playSlide = (index) => {
          const slide = carouselElement.querySelectorAll('.item')[index];
          const video = slide && slide.querySelector('video');
          if (video) {
            video.play().catch(e => {
              console.log('Video play prevented:', e);
            });
          }
        };

        carousel.on('show', (state) => playSlide(state.index));
        setTimeout(() => playSlide(0), 100);
      });
    }

    // Mobile: disable autoplay for carousel videos and add controls
    if (isMobile) {
      document.querySelectorAll(".results-carousel video").forEach(v => {
        v.autoplay = false;
        v.loop = true;
        v.muted = true;
        v.playsInline = true;
        v.preload = "metadata";
        v.controls = true;
        v.removeAttribute("autoplay");
      });
    }
    
    // Setup video autoplay for carousel (desktop) or view-based play/pause (mobile)
    setupVideoCarouselAutoplay();

})
