window.HELP_IMPROVE_VIDEOJS = false;

var INTERP_BASE = "./static/interpolation/";
var NUM_INTERP_FRAMES = 11;

var interp_images = [];
early = new Array(NUM_INTERP_FRAMES);
late = new Array(NUM_INTERP_FRAMES);

function preloadInterpolationImages() {
  for (var i = 0; i < NUM_INTERP_FRAMES; i++) {
    var early_path = INTERP_BASE + 'early/' + String(i) + '.jpg';
    early[i] = early_path;
    var late_path = INTERP_BASE + 'late/' + String(i) + '.jpg';
    late[i] = late_path;
  }
}

function setInterpolationImage_early(i) {
  var img = document.getElementById("img_early");
  img.src = early[i]+ "?r=" + Math.random();
}
function setInterpolationImage_late(i) {
  var img = document.getElementById("img_late");
  img.src = late[i]+ "?r=" + Math.random();
}

$(document).ready(function() {
    // Check for click events on the navbar burger icon
    $(".navbar-burger").click(function() {
      // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
      $(".navbar-burger").toggleClass("is-active");
      $(".navbar-menu").toggleClass("is-active");

    });

    var options = {
			slidesToScroll: 1,
			slidesToShow: 3,
			loop: true,
			infinite: true,
			autoplay: false,
			autoplaySpeed: 3000,
    }

		// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);

    // Loop on each carousel initialized
    for(var i = 0; i < carousels.length; i++) {
    	// Add listener to  event
    	carousels[i].on('before:show', state => {
    		console.log(state);
    	});
    }

    // Access to bulmaCarousel instance of an element
    var element = document.querySelector('#my-element');
    if (element && element.bulmaCarousel) {
    	// bulmaCarousel instance is available as element.bulmaCarousel
    	element.bulmaCarousel.on('before-show', function(state) {
    		console.log(state);
    	});
    }

    /*var player = document.getElementById('interpolation-video');
    player.addEventListener('loadedmetadata', function() {
      $('#interpolation-slider').on('input', function(event) {
        console.log(this.value, player.duration);
        player.currentTime = player.duration / 100 * this.value;
      })
    }, false);*/
    preloadInterpolationImages();

    $('#interpolation-slider').on('input', function(event) {
      setInterpolationImage_late(this.value);
      setInterpolationImage_early(this.value);
    });
    setInterpolationImage_early(0);
    setInterpolationImage_late(0);
    $('#interpolation-slider').prop('max', NUM_INTERP_FRAMES - 1);

    bulmaSlider.attach();

})
