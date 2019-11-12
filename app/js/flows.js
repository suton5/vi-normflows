/**
 * Flow functions
 *
 * - Identity
 * - Planar
 * - Radial
 */
IdentityFlow = function() {};

IdentityFlow.prototype.transform = function(data) {
  return data;
};

PlanarFlow = function(_w, _u, _b) {
  this.w0 = _w[0];
  this.w1 = _w[1];
  this.u0 = _u[0];
  this.u1 = _u[1];
  this.b = _b;
};
PlanarFlow.prototype.transform = function(z) {
  var flow = this;

  flow.w = [flow.w0, flow.w1];
  flow.u = [flow.u0, flow.u1];

  return z.map(function(d) {
    return math.add(d, math.multiply(flow.u, math.tanh(math.add(math.dot(flow.w, d), flow.b))));
  })
};
PlanarFlow.prototype.updateParams = function(newParams) {
  for (var param in newParams) {
    if (newParams.hasOwnProperty(param)) {
      this[param] = newParams[param];
    }
  }
};
