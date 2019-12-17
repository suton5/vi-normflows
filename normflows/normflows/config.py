from pathlib import Path

import autograd.numpy.random as npr


root = Path(__file__).parent.parent.parent
data = root / 'data'
notebooks = root / 'notebooks'
figs = root / 'figures'

figname = str(figs / 'iter_{}.png')
rs = npr.RandomState(101)