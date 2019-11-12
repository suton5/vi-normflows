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

var data = generateData(1000, 2, 0, 1);

var w = [0, 0],
    u = [0, 0],
    b = 0;

var noFlow = new PlanarFlow(w, u, b);

var gaussianDensity = new NormflowVis('vis-1', data, noFlow);


var testFlow = new PlanarFlow(w, u, b);

var planarFlow1 = new NormflowVis('vis-2', gaussianDensity.transformedData, testFlow, gaussianDensity);
