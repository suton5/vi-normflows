setTimeout(() => {

  MathJax.Hub.Config({
    tex2jax: {
      inlineMath: [ ['$','$'], ["\\(","\\)"] ],
      processEscapes: true
    }
  });
}, 1);

// Colors
var yellow = d3.interpolateYlGn(0), // "rgb(255, 255, 229)"
    yellowGreen = d3.interpolateYlGn(0.5), // "rgb(120, 197, 120)"
    green = d3.interpolateYlGn(1); // "rgb(0, 69, 41)"
function generateData(N, d, mu, sigma) {
  if (isNaN(mu)) {
    mu = 0;
  }
  if (isNaN(sigma)) {
    sigma = 1;

  }
  var normSamples = d3.randomNormal(mu, sigma);
  var data = [];
  d3.range(0, N).forEach(function() {
    obs = [];
    d3.range(0, d).forEach(function() {
      obs.push(normSamples(2));
    });
    data.push(obs);
  });
  return data;
}
function addMathJax(svg) {
  const continuation = () => {
    MathJax.Hub.Config({
      tex2jax: {
        inlineMath: [ ['$','$'], ["\\(","\\)"] ],
        processEscapes: true
      }
    });

    MathJax.Hub.Register.StartupHook("End", function() {
      setTimeout(() => {
        svg.selectAll('.tick').each(function(){
          var self = d3.select(this),
              g = self.select('text>span>svg');

          if(g[0][0] && g[0][0].tagName === 'svg') {
            g.remove();
            self.append(function(){
              return g.node();
            });
          }
        });
      }, 500);
    });

    MathJax.Hub.Queue(["Typeset", MathJax.Hub, svg.node()]);
  };

  wait((window.hasOwnProperty('MathJax')), continuation.bind(this));
}
function wait(condition, func, counter = 0) {
  if (condition || counter > 10) {
    return func()
  }

  setTimeout(wait.bind(null, condition, func, counter + 1), 30)
}

var data = generateData(1000, 2, 0, 1);

var w = [0, 0],
    u = [0, 0],
    b = 0;

var noFlow = new PlanarFlow(w, u, b);


var gaussianDensity = new NormflowVis('vis-1', data, noFlow);

var testFlow = new PlanarFlow(w, u, b);

var planarFlow1 = new NormflowVis('vis-2', gaussianDensity.transformedData, testFlow, gaussianDensity);

