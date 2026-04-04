import torch
import torch.nn as nn
import torch.nn.init as init
from typing import Literal
from jaxtyping import Float
from torch import Tensor




try:
    import tinycudann as tcnn
    TCNN_EXISTS = True
except ImportError:
    TCNN_EXISTS = False

def positional_encode_directions(ray_directions: torch.Tensor, L: int = 4) -> torch.Tensor:
    frequencies = 2.0 ** torch.arange(L, dtype=torch.float32, device=ray_directions.device)  # [L]
    angles = torch.pi * ray_directions.unsqueeze(-1) * frequencies  # [N, 3, L]
    sin_enc = torch.sin(angles)  # [N, 3, L]
    cos_enc = torch.cos(angles)  # [N, 3, L]
    encoded = torch.cat([sin_enc, cos_enc], dim=-1)  # [N, 3, 2*L]
    encoded = encoded.view(ray_directions.shape[0], -1)  # [N, 3*2*L] (展平)
    return encoded

class SHEncoding(nn.Module):
    """Spherical Harmonics encoding. Supports up to 4 levels (L=4, total 16 dimensions)."""

    def __init__(self, levels: int = 4, implementation: Literal["tcnn", "torch"] = "tcnn") -> None:
        super().__init__()
        if levels < 1 or levels > 4:
            raise ValueError(f"Spherical harmonic encoding only supports 1 to 4 levels, got {levels}")
        self.levels = levels
        self.output_dim = levels ** 2
        self.tcnn_encoding = None

        if implementation == "tcnn" and not TCNN_EXISTS:
            raise RuntimeError("Need tcnn! You must install tiny-cuda-nn")
        elif implementation == "tcnn":
            encoding_config = {"otype": "SphericalHarmonics","degree": levels}
            self.tcnn_encoding = tcnn.Encoding(
                n_input_dims=3,
                encoding_config=encoding_config,
            )

    def get_out_dim(self):
        return self.output_dim

    def forward(self, in_tensor: Float[Tensor, "*bs input_dim"]) -> Float[Tensor, "*bs output_dim"]:
        if self.tcnn_encoding is not None:
            return self.tcnn_encoding(in_tensor)
        return self.pytorch_fwd(in_tensor)

    def pytorch_fwd(self, in_tensor: Float[Tensor, "*bs input_dim"]) -> Float[Tensor, "*bs output_dim"]:
        """Forward pass using pytorch. Significantly slower than TCNN implementation."""
        return self.components_from_spherical_harmonics(dirs=in_tensor)

    def components_from_spherical_harmonics(self, dirs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dirs: Tensor of shape [N, 3], must be normalized direction vectors
        Returns:
            Tensor of shape [N, L^2] (e.g., L=4 => [N, 16])
        """


        x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
        result = []

        # L=0
        result.append(0.28209479177387814 * torch.ones_like(x))  # Y_0^0

        if self.levels >= 2:
            result += [
                -0.4886025119029199 * y,                    # Y_1^-1
                0.4886025119029199 * z,                     # Y_1^0
                -0.4886025119029199 * x                     # Y_1^1
            ]

        if self.levels >= 3:
            result += [
                1.0925484305920792 * x * y,                 # Y_2^-2
                -1.0925484305920792 * y * z,                # Y_2^-1
                0.31539156525252005 * (3 * z ** 2 - 1),      # Y_2^0
                -1.0925484305920792 * x * z,                # Y_2^1
                0.5462742152960396 * (x ** 2 - y ** 2),      # Y_2^2
            ]

        if self.levels >= 4:
            result += [
                0.5900435899266435 * y * (3 * x ** 2 - y ** 2),                  # Y_3^-3
                -1.7701307697799304 * x * y * z,                                # Y_3^-2
                0.9461746957575601 * y * (5 * z ** 2 - 1),                       # Y_3^-1
                -0.6690465435572892 * z * (5 * z ** 2 - 3),                      # Y_3^0
                0.9461746957575601 * x * (5 * z ** 2 - 1),                       # Y_3^1
                -0.8855003927832289 * z * (x ** 2 - y ** 2),                     # Y_3^2
                0.5900435899266435 * x * (x ** 2 - 3 * y ** 2),                  # Y_3^3
            ]

        if self.levels >= 5:
            raise NotImplementedError("SHEncoding supports up to level 4")

        return torch.stack(result, dim=-1)




class MLP(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        num_layers,  # 总层数（包含输入层和输出层）
        layer_width,
        activation=nn.ReLU(),
        out_activation=None, # 输出层激活函数（默认None，即不加）
        implementation="torch"
    ):
        super().__init__()
        self.implementation = implementation

        if implementation == "torch":
            layers = []
            for i in range(num_layers):
                in_features = in_dim if i == 0 else layer_width
                out_features = out_dim if i == num_layers - 1 else layer_width
                layers.append(nn.Linear(in_features, out_features))
                if i != num_layers - 1:
                    layers.append(activation)
                elif out_activation is not None:
                    layers.append(out_activation)
            self.model = nn.Sequential(*layers)

        elif implementation == "tcnn":
            import tinycudann as tcnn
            self.model = tcnn.NetworkWithInputEncoding(
                n_input_dims=in_dim,
                n_output_dims=out_dim,
                encoding_config={"otype": "Identity"},
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "Sigmoid",
                    "output_activation": "None" if out_activation is None else "Sigmoid",
                    "n_neurons": layer_width,
                    "n_hidden_layers": num_layers - 2,
                }
            )

    def forward(self, x):
        return self.model(x)


class Medium(nn.Module):
    def __init__(
        self,
        levels=4,
        hidden_dim=128,
        num_layers=4,
        density_bias=0.0,
        mlp_type="torch",  # "torch" or "tcnn"
    ):
        super().__init__()
        self.density_bias = density_bias
        self.medium_mlp = MLP(
            # in_dim=levels**2,
            in_dim = 16,
            out_dim=9,
            num_layers=num_layers,
            layer_width=hidden_dim,
            activation=nn.Sigmoid(),
            out_activation=None,
            implementation=mlp_type,
        )
        self.colour_activation = nn.Sigmoid()
        self.sigma_activation = nn.Softplus()

    def forward(self, dirs: torch.Tensor) -> torch.Tensor:
        return self.medium_mlp(dirs).float()   # [N, 9]

class Light(nn.Module):
    def __init__(
        self,
        levels=4,
        hidden_dim=128,
        num_layers=4,
        mlp_type="torch",  # "torch" or "tcnn"
    ):
        super().__init__()
        self.direction_encoding = SHEncoding(levels=levels)
        self.light_mlp = MLP(
            in_dim=self.direction_encoding.get_out_dim(),
            out_dim=3,
            num_layers=num_layers,
            layer_width=hidden_dim,
            activation=nn.Sigmoid(),
            out_activation=None,
            implementation=mlp_type,
        )
        self.output_activation = nn.Sigmoid()

    def forward(self, dirs: torch.Tensor) -> torch.Tensor:
        dirs_encoded = self.direction_encoding(dirs)
        return self.light_mlp(dirs_encoded).float()  # [N, 3]

def normalize(x, eps=1e-6):
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def schlick_fresnel(cos_theta, F0):
    return F0 + (1.0 - F0) * (1.0 - cos_theta).pow(5)


def ggx_distribution(normal, half_vec, roughness):
    alpha = roughness ** 2
    NdotH = torch.clamp((normal * half_vec).sum(-1, keepdim=True), 0.0, 1.0)
    denom = NdotH ** 2 * (alpha ** 2 - 1.0) + 1.0
    D = (alpha ** 2) / (torch.pi * denom ** 2 + 1e-6)
    return D


def smith_geometry(normal, view_dir, light_dir, roughness):
    def G1(v):
        NdotV = torch.clamp((normal * v).sum(-1, keepdim=True), 0.0, 1.0)
        k = (roughness + 1) ** 2 / 8.0
        return NdotV / (NdotV * (1 - k) + k + 1e-6)

    return G1(view_dir) * G1(light_dir)


class BRDF_old(nn.Module):
    def __init__(self, in_dim=13, hidden_dim=128, num_layers=3, out_dim=3):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_features = in_dim if i == 0 else hidden_dim
            out_features = out_dim if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_features, out_features))
            if i != num_layers - 1:
                layers.append(nn.ReLU(inplace=True))
            else:
                layers.append(nn.Sigmoid())  # RGB range [0,1]
        self.mlp = nn.Sequential(*layers)

    def forward(self, view_dir, normal, light_dir, roughness=None, specular=None):
        view_dir = normalize(view_dir)
        normal = normalize(normal)
        light_dir = normalize(light_dir)

        if roughness is None:
            roughness = torch.ones_like(view_dir[:, :1]) * 0.5
        if specular is None:
            specular = torch.ones_like(view_dir) * 0.04  # F0 default (dielectric)

        half_vec = normalize(view_dir + light_dir)
        NdotL = torch.clamp((normal * light_dir).sum(-1, keepdim=True), 0.0, 1.0)
        NdotV = torch.clamp((normal * view_dir).sum(-1, keepdim=True), 0.0, 1.0)
        VdotH = torch.clamp((view_dir * half_vec).sum(-1, keepdim=True), 0.0, 1.0)

        D = ggx_distribution(normal, half_vec, roughness)
        F = schlick_fresnel(VdotH, specular)
        G = smith_geometry(normal, view_dir, light_dir, roughness)

        spec = (D * F * G) / (4 * NdotL * NdotV + 1e-6)  # [N, 3]

        # build MLP input and predict albedo (diffuse term)
        brdf_input = torch.cat([normal, view_dir, light_dir, roughness, specular], dim=-1).to(view_dir.device)
        albedo = self.mlp(brdf_input)  # [N, 3]

        diffuse = albedo / torch.pi

        return NdotL * (diffuse + spec)  # final RGB  [N, 3]

class BRDF(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=3, implementation="torch"):
        super().__init__()
        self.implementation = implementation

        in_dim = 13
        out_dim = 3

        if implementation == "torch":
            layers = []
            for i in range(num_layers):
                in_features = in_dim if i == 0 else hidden_dim
                out_features = out_dim if i == num_layers - 1 else hidden_dim
                layers.append(nn.Linear(in_features, out_features))
                if i != num_layers - 1:
                    layers.append(nn.ReLU(inplace=True))
                else:
                    layers.append(nn.Sigmoid())  # RGB range [0,1]
            self.mlp = nn.Sequential(*layers)

        elif implementation == "tcnn":
            assert TCNN_EXISTS, "TCNN is not installed. Please install tiny-cuda-nn."
            self.mlp = tcnn.NetworkWithInputEncoding(
                n_input_dims=in_dim,
                n_output_dims=out_dim,
                encoding_config={"otype": "Identity"},
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "Sigmoid",
                    "n_neurons": hidden_dim,
                    "n_hidden_layers": num_layers - 2,
                }
            )

        else:
            raise ValueError(f"Unknown implementation: {implementation}")

    def forward(self, view_dir, normal, light_dir, roughness=None, specular=None):
        device = view_dir.device

        view_dir = normalize(view_dir).to(device)
        normal = normalize(normal).to(device)
        light_dir = normalize(light_dir).to(device)

        if roughness is None:
            roughness = torch.ones_like(view_dir[:, :1], device=device) * 0.5
        else:
            roughness = roughness.to(device)

        if specular is None:
            specular = torch.ones_like(view_dir, device=device) * 0.04  # F0 default (dielectric)
        else:
            specular = specular.to(device)

        half_vec = normalize(view_dir + light_dir).to(device)
        NdotL = torch.clamp((normal * light_dir).sum(-1, keepdim=True), 0.0, 1.0).to(device)
        NdotV = torch.clamp((normal * view_dir).sum(-1, keepdim=True), 0.0, 1.0).to(device)
        VdotH = torch.clamp((view_dir * half_vec).sum(-1, keepdim=True), 0.0, 1.0).to(device)

        D = ggx_distribution(normal, half_vec, roughness).to(device)
        F = schlick_fresnel(VdotH, specular).to(device)
        G = smith_geometry(normal, view_dir, light_dir, roughness).to(device)

        spec = (D * F * G) / (4 * NdotL * NdotV + 1e-6)  # [N, 3]

        brdf_input = torch.cat([normal, view_dir, light_dir, roughness, specular], dim=-1).to(device)
        albedo = self.mlp(brdf_input)
        diffuse = albedo / torch.pi

        return NdotL * (diffuse + spec)  # final RGB

def enhance_brdf(self,):

    ...
# class MLP(nn.Module):
#     def __init__(
#         self,
#         input_dim: int,
#         output_dim: int,
#         hidden_dims: list,
#         output_activation: str = None,
#         use_bias: bool = True,
#         init_method: str = "xavier",
#         device: str = "cuda"
#     ):
#         super().__init__()
#         layers = []
#         prev_dim = input_dim
#
#
#         for hidden_dim in hidden_dims:
#             layers.append(nn.Linear(prev_dim, hidden_dim, bias=use_bias))
#             self._init_weights(layers[-1], init_method)
#             prev_dim = hidden_dim
#
#         layers.append(nn.Linear(prev_dim, output_dim, bias=use_bias))
#         if output_activation:
#             layers.append(self._get_activation(output_activation))
#
#         self.net = nn.Sequential(*layers).to(device)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.net(x)
#
#     def _get_activation(self, name: str) -> nn.Module:
#
#         activations = {
#             "relu": nn.ReLU(inplace=True),
#             "leaky_relu": nn.LeakyReLU(0.2, inplace=True),
#             "sigmoid": nn.Sigmoid(),
#             "tanh": nn.Tanh(),
#         }
#         return activations[name.lower()]
#
#     def _init_weights(self, layer: nn.Linear, method: str):
#
#         if method == "xavier":
#             init.xavier_uniform_(layer.weight)
#         elif method == "kaiming":
#             init.kaiming_uniform_(layer.weight, mode="fan_in", nonlinearity="relu")
#         elif method == "normal":
#             init.normal_(layer.weight, mean=0.0, std=0.02)
#         if layer.bias is not None:
#             init.zeros_(layer.bias)
#
#     def __repr__(self):
#         return f"MLP(input_dim={self.net[0].in_features}, output_dim={self.net[-1].out_features}, hidden_dims={[layer.out_features for layer in self.net if isinstance(layer, nn.Linear)][:-1]})"