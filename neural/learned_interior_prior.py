"""Proposes interior exits from a directional table, as an optional addition to the operator.

This gives the renderer a second source of exit samples beside the operator. It uses a radiance distribution map
(RDM) table, and the renderer selects it with `--rdm_prior`.

The table holds exit directions for each entry direction, for paths that entered and escaped. It must be built from
the training split of a pool. It ignores position and wavelength, so importance weights keep the operator's learned
target. Therefore, this changes how exits are sampled, and it does not move the target closer to the physics.
"""
import math
import torch


# A table of exit directions for each entry direction, which can sample exits and return their density.
class LearnedInteriorPrior:
    def __init__(self, pmf, theta_bins, phi_bins, model):
        self.pmf = torch.as_tensor(pmf, dtype=torch.float32)
        self.theta_bins, self.phi_bins = int(theta_bins), int(phi_bins)
        self.model = model
        edges = torch.linspace(0, math.pi, self.theta_bins+1)
        self.solid_angle = ((edges[:-1].cos()-edges[1:].cos())[:,None].expand(-1,self.phi_bins)*2*math.pi/self.phi_bins).flatten()
        t = model.triangles
        self.area = torch.linalg.vector_norm(torch.cross(t[:,1]-t[:,0],t[:,2]-t[:,0],dim=-1),dim=-1)/2
        size = self.theta_bins*self.phi_bins
        if self.pmf.shape != (size,size) or not torch.isfinite(self.pmf).all() or not (self.pmf>0).all():
            raise ValueError('Prior must be a finite, positive square directional PMF')
        if not torch.allclose(self.pmf.sum(-1),torch.ones(size),atol=1e-5):
            raise ValueError('Prior rows must sum to one')

    # Returns the table index of each direction.
    def bins(self, direction):
        theta = torch.acos(direction[:,2].clamp(-1,1))
        phi = torch.atan2(direction[:,1],direction[:,0])
        ti = (theta/math.pi*self.theta_bins).long().clamp(0,self.theta_bins-1)
        pi = ((phi+math.pi)/(2*math.pi)*self.phi_bins).long().clamp(0,self.phi_bins-1)
        return ti*self.phi_bins+pi

    # Returns the log density of an exit direction and facet under this table.
    def log_pdf(self, x, facet, direction):
        row, column = self.bins(-x[:,3:6]), self.bins(direction)
        cosine = (direction @ self.model.normal.T).clamp_min(0)
        projected_area = (cosine*self.area).sum(-1)
        # The facet is chosen by projected area, and the point is then uniform over the facet.
        cosine_f = cosine.gather(1,facet[:,None]).squeeze(-1)
        return (self.pmf[row,column]/self.solid_angle[column]).log()+cosine_f.log()-projected_area.log()

    # Samples an exit direction from the table, then a facet by projected area, then a uniform point on that facet.
    def sample(self, x, generator=None):
        n = len(x)
        row = self.bins(-x[:,3:6])
        cell = torch.multinomial(self.pmf[row],1,generator=generator).squeeze(-1)
        u = torch.rand((n,4),generator=generator)
        lo = (cell//self.phi_bins)*math.pi/self.theta_bins
        hi = lo+math.pi/self.theta_bins
        c = (1-u[:,0])*lo.cos()+u[:,0]*hi.cos()
        phi = -math.pi+((cell%self.phi_bins)+u[:,1])*2*math.pi/self.phi_bins
        sint = (1-c.square()).clamp_min(0).sqrt()
        direction = torch.stack([sint*phi.cos(),sint*phi.sin(),c],-1)
        facet_mass = (direction @ self.model.normal.T).clamp_min(0)*self.area
        facet = torch.multinomial(facet_mass,1,generator=generator).squeeze(-1)
        root = u[:,2].sqrt()
        bary = torch.stack([1-root,root*(1-u[:,3]),root*u[:,3]],-1)
        position = (self.model.triangles[facet]*bary[:,:,None]).sum(1)
        return facet, position, direction


# Draws each exit from the table with probability `alpha`, and from the operator otherwise. Importance weights keep the
# operator's target. `alpha` must stay below 1, so that the operator can still produce every exit it supports.
def sample_mixture(model, prior, x, alpha, generator=None):
    if not 0 <= alpha < 1:
        raise ValueError('Prior fraction must be in [0,1) to retain neural support')
    result = model.sample(x,generator)
    if alpha == 0:
        return result
    use_prior = torch.rand(len(x),generator=generator) < alpha
    facet, position, direction = prior.sample(x,generator)
    for key, values in [('exit_facet',facet),('exit_position',position),('exit_direction',direction)]:
        mask = use_prior if values.ndim == 1 else use_prior[:,None]
        result[key] = torch.where(mask,values,result[key])
    log_target = model.log_exit_pdf(x,result['exit_facet'],result['exit_position'],result['exit_direction'])
    log_prior = prior.log_pdf(x,result['exit_facet'],result['exit_direction'])
    log_proposal = torch.logaddexp(log_target+math.log1p(-alpha),log_prior+math.log(alpha))
    weight = (log_target-log_proposal).exp()
    if not torch.isfinite(weight).all():
        raise RuntimeError('Nonfinite learned interior importance weight')
    result['throughput'] = result['escaped'].float()*weight
    # The barycentrics describe the operator's samples, not the table's, so they are dropped.
    result.pop('barycentric',None)
    return result
