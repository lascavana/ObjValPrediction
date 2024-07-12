import ecole
import numpy as np
import pyscipopt as scip

def integrate(values):
    assert(len(values)>1)

    # normalize #
    values = np.sort(values)
    values = values - values[0]
    if values[-1] > 1e-5:
        values = values / values[-1]

    num_samples = values.shape[0]
    l = 1/(num_samples-1)

    integral = 0.0
    for j in range(num_samples-1):
        integral += (values[j] + values[j+1]) * l/2

    return integral

def estimate_linear_trend(x):
    indices = np.arange(len(x))
    trend, _ = np.polyfit(indices, x, 1)
    return trend

class RootBoundReward(ecole.reward.IsDone):
    def extract(self, model, done):
        m = model.as_pyscipopt()
        a = m.getLPObjVal()
        return a

class SolveStats:
    def __init__(self):
        self.reset()

    def reset(self):
        self.gap = []
        self.tree_weight = []
        self.leaf_freq = []
        self.profile = []
        self.onodes = []
        self.primal = []
        self.dual = []
        self.median = []
        self.mgap = []
        self.estimate = []
        self.size_rank1 = []

class Predictor:
    def __init__(self):
        self.reset()
    def reset(self):
        self.prediction = False


class EventHandler(scip.Eventhdlr):
    """
    A SCIP event handler that records solving stats
    """
    def __init__(self, seed=0, sample_prob=0.05, window_size=100):
        self.tree_weight = 0.0
        self.first_dual = None
        self.first_primal = None
        self.stats = SolveStats()
        self.bestestimate_per_depth = {}
        self.window_size = window_size
        
        # sampling #
        self.sample_prob = sample_prob
        self.rng = np.random.RandomState(seed)
        self.samples = []

        # internal predictors #
        self.predictor_estim = Predictor()
        self.predictor_rank1 = Predictor()

        self.callcount = 0
        
    def reset(self):
        self.tree_weight = 0.0
        self.first_dual = None
        self.first_primal = None
        self.bestestimate_per_depth = {}
        self.stats.reset()

        self.predictor_estim.reset()
        self.predictor_rank1.reset()

        self.samples = []
        self.callcount = 0
        

    def eventinit(self):
        self.model.catchEvent(scip.SCIP_EVENTTYPE.NODEBRANCHED, self)
        self.model.catchEvent(scip.SCIP_EVENTTYPE.NODEFEASIBLE, self)
        self.model.catchEvent(scip.SCIP_EVENTTYPE.NODEINFEASIBLE, self)

    def eventexit(self):
        self.model.dropEvent(scip.SCIP_EVENTTYPE.NODEBRANCHED, self)
        self.model.dropEvent(scip.SCIP_EVENTTYPE.NODEFEASIBLE, self)
        self.model.dropEvent(scip.SCIP_EVENTTYPE.NODEINFEASIBLE, self)

    def eventexec(self, event):
        event_type = event.getType()
        if event_type == scip.SCIP_EVENTTYPE.NODEFEASIBLE or event_type == scip.SCIP_EVENTTYPE.NODEINFEASIBLE:
            self.update_tree_weight()
        self.update_bestestimate_per_depth()
            
        # first primal #
        primal = self.model.getUpperbound()
        if self.first_primal is None and primal < 1e8:
            self.first_primal = primal
            self.first_dual = self.model.getLowerbound()

        # update saved statistics #
        self.update_stats()

        # sample collection #
        if (self.callcount < 100) or (primal > 1e8):  # no collection in first 100 nodes
            self.callcount += 1
        else:
            collect_sample = self.rng.rand() < self.sample_prob
            if collect_sample:
                sample = self.collect_sample()
                self.samples.append(sample)

    def collect_sample(self):
        attributes = self.stats.__dict__

        w1 = self.window_size
        w2 = int(self.window_size/2)

        sample = {}
        for key, val in attributes.items():
            value = val[-1]
            trend1 = estimate_linear_trend( val[-w1:] ) 
            trend2 = estimate_linear_trend( val[-w2:] ) 
            variance1 = np.var( val[-w1:] )
            variance2 = np.var( val[-w2:] )
            sample[key] = [value, trend1, variance1, trend2, variance2]

        sample["incumbent"] = self.model.getUpperbound()
        sample["predictor_estim"] = self.predictor_estim.prediction
        sample["predictor_rank1"] = self.predictor_rank1.prediction

        return sample

    def update_tree_weight(self):
        depth = self.model.getDepth()
        self.tree_weight += 2**(-depth)

    def update_bestestimate_per_depth(self):
        node =  self.model.getCurrentNode()
        d = node.getDepth()
        c = node.getEstimate()

        if d in self.bestestimate_per_depth:
            if c < self.bestestimate_per_depth[d]: 
                self.bestestimate_per_depth[d] = c 
        else:
            self.bestestimate_per_depth[d] = c

    def update_stats(self):
        leaves, children, siblings = self.model.getOpenNodes()
        onodes = leaves + children + siblings
        onode_bounds = [n.getLowerbound() for n in onodes]
        onode_estimates = [n.getEstimate() for n in onodes]
        
        # ensure we have not reached the end of the search #
        if len(onodes) == 0:
            return

        # gap #
        self.stats.gap.append( self.model.getGap() )

        # tree_weight #
        self.stats.tree_weight.append( self.tree_weight )
               
        # leaf frequency #
        F = self.model.getNFeasibleLeaves() + self.model.getNInfeasibleLeaves()
        k = self.model.getNNodes()
        leaf_freq = (F-0.5)/k
        self.stats.leaf_freq.append( leaf_freq )

        # open nodes #
        self.stats.onodes.append( self.model.getNLeaves() )

        # profile #
        if len(onode_bounds) > 1:
            profile = integrate(onode_bounds)
        else:
            profile = 0.0
        self.stats.profile.append(profile)

        # primal #
        primal = self.model.getUpperbound() 
        self.stats.primal.append( primal / self.first_primal )

        # dual #
        dual = self.model.getLowerbound() 
        self.stats.dual.append( dual / self.first_primal )

        # median #
        median = np.median(onode_bounds) 
        self.stats.median.append( median / self.first_primal )

        # median gap #
        mgap = abs(primal - median)
        mgap /= abs(self.first_primal - self.first_dual)
        self.stats.mgap.append( mgap )

        # estimate #
        estimate = np.amin(onode_estimates) 
        self.stats.estimate.append( estimate / self.first_primal )

        # rank 1 #
        size_rank1 = 0
        for node in onodes:
            d = node.getDepth()
            c = node.getEstimate()

            if d in self.bestestimate_per_depth:
                if c <= self.bestestimate_per_depth[d]: 
                    size_rank1 += 1
            else:
                size_rank1 += 1
        self.stats.size_rank1.append( size_rank1 )

        # update internal predictors #
        if (not self.predictor_estim.prediction) and (primal <= estimate):
            self.predictor_estim.prediction = True
        if (not self.predictor_rank1.prediction) and (size_rank1 == 0):
            self.predictor_rank1.prediction = True


