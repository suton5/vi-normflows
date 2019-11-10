# ToDo
Sun. Nov. 10

1. Constructed posterior
    1. Start with a mean field gaussian
    2. Apply $K$ hard-coded planar flows (ensure normalising)
    3. Implement gradient descent and see if the model can match the posterior AND match the parameters of the planar flows. Note that algorithm should only be optimising over $\phi$. Let us implement $\mathcal{F}$ without $\beta_t$ first.
2. Their posterior
    1. Implement one of the posteriors in Table 1
    2. Implement gradient descent and try to match the posterior with comparable numbers of flows as the authors used (recreate fig. 3). Note that algorithm should only be optimising over $\phi$. Let us implement $\mathcal{F}$ without $\beta_t$ first.
3. The posterior for a very simple model
    1. Assume $f_{\theta}$ is a simple $Az_n+b$. Assign our own $A,b$.
    2. Create fake $z_n$, get corresponding $x_n$ (generative process).
    3. Do the algorithm
    4. Check if inferred $z_n$ from posterior matches fake $z_n$. Also check if inferred $A,b$ matches our set $A,b$.