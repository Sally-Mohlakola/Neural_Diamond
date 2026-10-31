"""Predicts how the other wavelengths of a path leave the diamond, given the exit of its hero wavelength.

Dispersion splits white light into the colours known as fire. The operator predicts one exit, for the hero wavelength.
Therefore, this optional head, selected in the renderer with `--dispersion`, models the others so that they can leave
differently. For each other wavelength, a gate decides whether it stays on the hero's route, and a shift says how far
it moves. A wavelength that leaves the route gets its own draw from the operator. Both outcomes are sampled, so the
head adds no importance weights to the renderer.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from neural.learned_interior_model import frames

# These map the visible range of 360 to 830 nm onto -1 to 1, as the operator's inputs do.
WAVELENGTH_CENTRE, WAVELENGTH_SCALE = 595., 235.


# Returns the Sellmeier refractive index of diamond, with the same fit as `bsdf.dispersion.diamond_ior`. It is plain
# arithmetic so that the head can run without a Mitsuba variant.
# The index gap between two colours drives the shift, not their wavelength gap. The index is steep in the blue and flat
# in the red, so equal wavelength gaps disperse very differently.
def diamond_index(wavelength_nm):
    l2 = (wavelength_nm/1000.)**2
    return (1 + .3306*l2/(l2 - .175**2) + 4.3356*l2/(l2 - .106**2))**.5


# Returns the exit coordinates in the operator's convention, which is the inverse of `decode`. It also flags exits on a
# facet edge or at a grazing angle, where the coordinates are clamped to stay finite.
def coordinates(triangles, tangent, bitangent, normal, facet, position, direction):
    tri = triangles[facet]
    edges = tri[:, :2] - tri[:, 2:3]
    delta = position - tri[:, 2]
    gram = edges @ edges.transpose(1, 2)
    uv = torch.linalg.solve(gram, (edges @ delta[..., None])).squeeze(-1)
    bary = torch.cat([uv, 1 - uv.sum(-1, keepdim=True)], -1)
    cosine = (direction*normal[facet]).sum(-1)
    clipped = (bary.min(-1).values < 1e-5) | (cosine < 1e-5)
    bary = bary.clamp_min(1e-5)
    cosine = cosine.clamp_min(1e-5)
    y = torch.stack([torch.log(bary[:, 0]/bary[:, 2]), torch.log(bary[:, 1]/bary[:, 2]),
                     (direction*tangent[facet]).sum(-1)/cosine,
                     (direction*bitangent[facet]).sum(-1)/cosine], -1)
    return y, clipped


# Returns the surface point and outward unit direction of exit coordinates, exactly as the operator decodes them.
def decode(triangles, tangent, bitangent, normal, facet, y):
    bary = torch.cat([y[:, :2], torch.zeros_like(y[:, :1])], -1).softmax(-1)
    position = (triangles[facet]*bary[:, :, None]).sum(1)
    direction = F.normalize(normal[facet] + y[:, 2:3]*tangent[facet]
                            + y[:, 3:4]*bitangent[facet], dim=-1)
    return position, direction


# The gate and shift for one other wavelength, given the hero's exit.
# The inputs are the operator's ten entry features, a facet embedding, the hero's exit coordinates, the other
# wavelength, the wavelength gap, and the index gap. The exit coordinates pass through asinh because slopes grow large
# at grazing exits. The index gap is scaled so that diamond's visible dispersion of about 0.05 spans about five units.
# Shifts are standardised per coordinate, with scales stored in the checkpoint, so that all four are learned equally.
class DispersionModel(nn.Module):
    def __init__(self, vertices, faces, width=128, depth=4):
        super().__init__()
        tri, t, b, n = frames(vertices, faces)
        for name, value in [('triangles', tri), ('tangent', t), ('bitangent', b), ('normal', n)]:
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float32))
        self.register_buffer('shift_scale', torch.ones(4))
        self.width, self.depth = width, depth
        self.embedding = nn.Embedding(len(faces), 16)
        layers, size = [], 10 + 16 + 4 + 3
        for _ in range(depth - 1):
            layers += [nn.Linear(size, width), nn.SiLU()]
            size = width
        self.trunk = nn.Sequential(*layers)
        self.gate = nn.Linear(width, 1)
        self.shift = nn.Linear(width, 4)

    # Returns the gate logit and the shift. `x` holds the operator's features at the hero wavelength, `y` holds the
    # hero's exit coordinates, and `wavelength` is the other wavelength in nm.
    def forward(self, x, facet, y, wavelength):
        hero = x[:, 9:10]
        other = ((wavelength - WAVELENGTH_CENTRE)/WAVELENGTH_SCALE)[:, None]
        # Wavelengths are clamped to the visible range because the Sellmeier fit has poles in the UV.
        index_gap = 100*(diamond_index(wavelength[:, None].clamp(360., 830.))
                         - diamond_index((hero*WAVELENGTH_SCALE + WAVELENGTH_CENTRE).clamp(360., 830.)))
        h = self.trunk(torch.cat([x, self.embedding(facet), torch.asinh(y), other, other - hero,
                                  index_gap], -1))
        return self.gate(h).squeeze(-1), self.shift(h)*self.shift_scale

    # Returns the gate and shift losses. The gate is scored on every pair. The shift is scored only on `usable` pairs,
    # which are those that stay on the route and were not clamped at an edge or a grazing exit.
    def losses(self, x, facet, y, wavelength, stays, shift, usable):
        logit, predicted = self(x, facet, y, wavelength)
        gate_loss = F.binary_cross_entropy_with_logits(logit, stays.float())
        if not usable.any():
            return gate_loss, gate_loss*0
        shift_loss = F.smooth_l1_loss(predicted[usable]/self.shift_scale,
                                      shift[usable]/self.shift_scale)
        return gate_loss, shift_loss

    # Samples whether each wavelength stays on the hero's route, and gives its exit on the hero's facet.
    @torch.no_grad()
    def sample(self, x, facet, y, wavelength, generator=None):
        logit, shift = self(x, facet, y, wavelength)
        stays = torch.rand(len(x), generator=generator) < torch.sigmoid(logit)
        position, direction = decode(self.triangles, self.tangent, self.bitangent,
                                     self.normal, facet, y + shift)
        return dict(stays=stays, exit_position=position, exit_direction=direction,
                    stay_probability=torch.sigmoid(logit))


# Returns the exit coordinates of an operator sample, to pass to the dispersion head.
def hero_coordinates(model, facet, position, direction):
    y, _ = coordinates(model.triangles, model.tangent, model.bitangent, model.normal,
                       facet, position, direction)
    return y


# Loads a trained dispersion head on the CPU. If an operator checkpoint is given, it checks that both use the same mesh.
def load_dispersion(path, operator_checkpoint=None):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('schema_version') != 1 or checkpoint.get('kind') != 'dispersion':
        raise ValueError('Not a dispersion checkpoint: %s' % path)
    if operator_checkpoint is not None and not torch.equal(
            checkpoint['faces'], operator_checkpoint['faces'].to(checkpoint['faces'].dtype)):
        raise ValueError('Dispersion head and learned interior operator use different geometry')
    model = DispersionModel(checkpoint['vertices'].numpy(), checkpoint['faces'].numpy(),
                            checkpoint['width'], checkpoint['depth'])
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model, checkpoint
