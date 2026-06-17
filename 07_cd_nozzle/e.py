import json

path = "cd-nozzle-addpinn-unsteady.ipynb"
nb = json.load(open(path))


# ── helpers ───────────────────────────────────────────────────────────────────
def get_src(cell):
    return "".join(cell["source"])


def set_src(cell, s):
    cell["source"] = [l + "\n" for l in s.split("\n")]


# ──────────────────────────────────────────────────────────────────────────────
# PATCH 1  (cell-03  UnsteadyNet)
# BUG:  hard-pressure constraint  x*(x-1)*p_raw  produces NaN when |p_raw|
#       is large because the quadratic can push P negative before softplus
#       has had a chance to bound it, and sqrt(GAMMA*P/rho) then gives NaN.
# FIX:  (a) Remove hard-pressure from the network output — keep it soft only
#            in the loss (the IC already pins P=1 at t=0 strongly).
#       (b) Add a small eps inside every sqrt(P/rho) call in the loss to
#            prevent NaN gradients.
# ──────────────────────────────────────────────────────────────────────────────
OLD_NET = '''\
    def forward(self, xt: torch.Tensor) -> torch.Tensor:
        """xt: (N, 2) columns [x, t]. Returns (N, 4): [rho, u, P, T]."""
        out = self.net(xt)
        x   = xt[:, 0:1]

        rho_raw, u_raw, p_raw, T_raw = (
            out[:, 0:1], out[:, 1:2], out[:, 2:3], out[:, 3:4]
        )

        rho = nn.functional.softplus(rho_raw)           # strictly positive
        u   = u_raw                                      # unconstrained
        # hard pressure constraint: P(0)=1, P(1)=P_EXIT always
        P   = x * (x - 1.0) * p_raw + (1.0 - (1.0 - P_EXIT) * x)
        T   = nn.functional.softplus(T_raw)             # strictly positive

        return torch.cat([rho, u, P, T], dim=1)'''

NEW_NET = '''\
    def forward(self, xt: torch.Tensor) -> torch.Tensor:
        """xt: (N, 2) columns [x, t]. Returns (N, 4): [rho, u, P, T]."""
        out = self.net(xt)

        rho_raw, u_raw, p_raw, T_raw = (
            out[:, 0:1], out[:, 1:2], out[:, 2:3], out[:, 3:4]
        )

        # softplus keeps rho and P strictly positive and bounded away from 0,
        # preventing NaN in sqrt(gamma*P/rho) during the first Adam epochs.
        # Bias offset of +1 means softplus(0)~0.69 → initialises near 1.
        rho = nn.functional.softplus(rho_raw + 1.0)
        u   = u_raw
        P   = nn.functional.softplus(p_raw   + 1.0)
        T   = nn.functional.softplus(T_raw   + 1.0)

        return torch.cat([rho, u, P, T], dim=1)'''

for cell in nb["cells"]:
    src = get_src(cell)
    if "hard pressure constraint" in src and "P   = x * (x - 1.0)" in src:
        set_src(cell, src.replace(OLD_NET, NEW_NET))
        print("PATCH 1 applied: UnsteadyNet.forward")
        break

# ──────────────────────────────────────────────────────────────────────────────
# PATCH 2  (cell-05  UnsteadyLoss.pde + bc)
# BUG 1:  sqrt(GAMMA * P / rho) inside interface() and pde() gives NaN
#         gradient when rho or P is tiny — even if the value is finite,
#         d/d(P) of sqrt(P/rho) → ∞ as P→0.
# FIX:    add EPS=1e-6 inside every sqrt to clip gradient magnitude.
#
# BUG 2:  bc() uses hard coded P_EXIT for the exit, but we removed the hard
#         constraint from the network, so we need a soft exit-P BC now.
# FIX:    add outlet P residual back into bc().
# ──────────────────────────────────────────────────────────────────────────────
OLD_PDE = '''\
    @staticmethod
    def pde(model: UnsteadyNet, xt: torch.Tensor) -> torch.Tensor:
        """Returns (N, 4) residual tensor [mass, mom, energy, ideal]."""
        out = model(xt)
        rho, u, P, T = out[:, 0:1], out[:, 1:2], out[:, 2:3], out[:, 3:4]
        x   = xt[:, 0:1]
        A   = get_area(x)
        A_x = get_d_area(x)

        rho_x, rho_t = _grad(rho, xt)
        u_x,   u_t   = _grad(u,   xt)
        P_x,   _     = _grad(P,   xt)
        T_x,   T_t   = _grad(T,   xt)

        mass     = A * rho_t + A * u * rho_x + A * rho * u_x + rho * u * A_x
        momentum = A * (rho * (u_t + u * u_x) + P_x)
        energy   = rho * A * (T_t + u * T_x) + (GAMMA - 1.0) * P * (A * u_x + A_x * u)
        ideal    = P - rho * T

        return torch.cat([mass, momentum, energy, ideal], dim=1)

    @staticmethod
    def bc(model: UnsteadyNet, xt: torch.Tensor) -> torch.Tensor:
        """
        Soft BCs for rho and u (pressure is handled by hard constraint).
        Returns scalar.

        Inlet (x≈0):  rho=1, u=U_INLET, T=1
        IC    (t≈0):  rho=1, u=0, P=1, T=1
        """
        out  = model(xt)
        rho, u, P, T = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
        x, t = xt[:, 0], xt[:, 1]

        # inlet
        in_mask = x < BD_DELTA
        inlet_loss = (
            (rho[in_mask] - 1.0).pow(2).mean()
            + (u[in_mask]   - U_INLET).pow(2).mean()
            + (T[in_mask]   - 1.0).pow(2).mean()
        ) if in_mask.any() else torch.tensor(0.0, device=device)

        # IC at t=0: at rest, uniform state
        ic_mask = t < BD_DELTA
        ic_loss = (
            (rho[ic_mask] - 1.0).pow(2).mean()
            + (u[ic_mask]  - 0.0).pow(2).mean()
            + (P[ic_mask]  - 1.0).pow(2).mean()
            + (T[ic_mask]  - 1.0).pow(2).mean()
        ) if ic_mask.any() else torch.tensor(0.0, device=device)

        return inlet_loss + ic_loss'''

NEW_PDE = '''\
    EPS = 1e-6   # prevents NaN gradients in sqrt(P/rho) when near zero

    @staticmethod
    def pde(model: UnsteadyNet, xt: torch.Tensor) -> torch.Tensor:
        """Returns (N, 4) residual tensor [mass, mom, energy, ideal]."""
        out = model(xt)
        rho, u, P, T = out[:, 0:1], out[:, 1:2], out[:, 2:3], out[:, 3:4]
        x   = xt[:, 0:1]
        A   = get_area(x)
        A_x = get_d_area(x)

        rho_x, rho_t = _grad(rho, xt)
        u_x,   u_t   = _grad(u,   xt)
        P_x,   _     = _grad(P,   xt)
        T_x,   T_t   = _grad(T,   xt)

        mass     = A * rho_t + A * u * rho_x + A * rho * u_x + rho * u * A_x
        momentum = A * (rho * (u_t + u * u_x) + P_x)
        energy   = rho * A * (T_t + u * T_x) + (GAMMA - 1.0) * P * (A * u_x + A_x * u)
        ideal    = P - rho * T

        return torch.cat([mass, momentum, energy, ideal], dim=1)

    @staticmethod
    def bc(model: UnsteadyNet, xt: torch.Tensor) -> torch.Tensor:
        """
        Soft BCs — all boundaries enforced as MSE penalties.
        Returns scalar.

        Inlet  (x≈0):  rho=1, u=U_INLET, P=1, T=1
        Outlet (x≈1):  P=P_EXIT
        IC     (t≈0):  rho=1, u=0, P=1, T=1
        """
        out  = model(xt)
        rho, u, P, T = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
        x, t = xt[:, 0], xt[:, 1]

        zero = torch.tensor(0.0, device=device)

        # inlet x≈0
        in_mask = x < BD_DELTA
        inlet_loss = (
            (rho[in_mask] - 1.0    ).pow(2).mean()
            + (u[in_mask] - U_INLET).pow(2).mean()
            + (P[in_mask] - 1.0    ).pow(2).mean()
            + (T[in_mask] - 1.0    ).pow(2).mean()
        ) if in_mask.any() else zero

        # outlet x≈1
        out_mask = x > 1.0 - BD_DELTA
        outlet_loss = (
            (P[out_mask] - P_EXIT).pow(2).mean()
        ) if out_mask.any() else zero

        # IC at t=0
        ic_mask = t < BD_DELTA
        ic_loss = (
            (rho[ic_mask] - 1.0).pow(2).mean()
            + (u[ic_mask] - 0.0 ).pow(2).mean()
            + (P[ic_mask] - 1.0 ).pow(2).mean()
            + (T[ic_mask] - 1.0 ).pow(2).mean()
        ) if ic_mask.any() else zero

        return inlet_loss + outlet_loss + ic_loss'''

for cell in nb["cells"]:
    src = get_src(cell)
    if "Soft BCs for rho and u (pressure is handled by hard constraint)" in src:
        set_src(cell, src.replace(OLD_PDE, NEW_PDE))
        print("PATCH 2 applied: UnsteadyLoss.pde + bc")
        break

# ──────────────────────────────────────────────────────────────────────────────
# PATCH 3  (cell-10  DualUnsteadyLoss.interface)
# BUG:  x_s.expand(t_pts.shape[0], 1) when x_s is a 1-D clamped parameter
#       creates a tensor whose grad_fn goes through clamp(), and then
#       requires_grad_(True) on xt_i creates a double-autograd path that
#       causes NaN in the second backward (shock_opt step).
# FIX:  detach x_s before building xt_i so that gradients w.r.t. x_shock
#       flow only through the flux difference terms (rh_mass etc.), not
#       through the coordinate itself.  Also add EPS inside every sqrt.
# ──────────────────────────────────────────────────────────────────────────────
OLD_IFACE = """\
        x_s = shock.clamped_x()                # scalar
        # shock speed (autograd through x_shock parameter)
        # approximate W ≈ 0 for pseudo-time; keeps formulation simple and
        # reduces to steady RH at convergence
        W = torch.zeros(1, device=device)

        # build (N,2) xt tensors at the interface
        x_col = x_s.expand(t_pts.shape[0], 1)
        t_col = t_pts[:, None]
        xt_i  = torch.cat([x_col, t_col], dim=1).requires_grad_(True)

        out_L = net_L(xt_i)
        out_R = net_R(xt_i)
        rho_L, u_L, P_L = out_L[:, 0], out_L[:, 1], out_L[:, 2]
        rho_R, u_R, P_R = out_R[:, 0], out_R[:, 1], out_R[:, 2]
        A_i   = get_area(x_s).expand_as(rho_L)

        H_L = GAMMA / (GAMMA - 1.0) * P_L / rho_L
        H_R = GAMMA / (GAMMA - 1.0) * P_R / rho_R

        flux_L_mass = rho_L * (u_L - W) * A_i
        flux_R_mass = rho_R * (u_R - W) * A_i
        flux_L_mom  = (rho_L * (u_L - W) ** 2 + P_L) * A_i
        flux_R_mom  = (rho_R * (u_R - W) ** 2 + P_R) * A_i
        flux_L_en   = rho_L * (u_L - W) * (H_L + 0.5 * (u_L - W) ** 2) * A_i
        flux_R_en   = rho_R * (u_R - W) * (H_R + 0.5 * (u_R - W) ** 2) * A_i

        rh_mass = (flux_L_mass - flux_R_mass).pow(2).mean()
        rh_mom  = (flux_L_mom  - flux_R_mom ).pow(2).mean()
        rh_en   = (flux_L_en   - flux_R_en  ).pow(2).mean()

        a_L = torch.sqrt(GAMMA * P_L / rho_L)
        a_R = torch.sqrt(GAMMA * P_R / rho_R)
        sup_pre  = torch.relu(a_L - u_L).pow(2).mean()   # penalise u_L < a_L
        sub_post = torch.relu(u_R - a_R).pow(2).mean()   # penalise u_R > a_R
        entropy  = torch.relu(P_L - P_R).pow(2).mean()   # penalise P_L > P_R (wrong entropy)

        return rh_mass + rh_mom + rh_en + sup_pre + sub_post + entropy"""

NEW_IFACE = """\
        x_s = shock.clamped_x()                # scalar, keeps grad for shock_opt

        # Detach x_s when building the coordinate tensor so the autograd graph
        # through xt_i (needed for PDE spatial derivatives) stays separate from
        # the graph through x_s (needed for shock parameter gradients).
        # Shock-position gradients enter only through the flux-difference terms.
        x_s_val = x_s.detach()
        x_col   = x_s_val.expand(t_pts.shape[0]).unsqueeze(1)   # (N,1)
        t_col   = t_pts.unsqueeze(1)                             # (N,1)
        # No requires_grad here — the networks handle their own graph
        xt_i    = torch.cat([x_col, t_col], dim=1)

        out_L = net_L(xt_i)
        out_R = net_R(xt_i)
        rho_L, u_L, P_L = out_L[:, 0], out_L[:, 1], out_L[:, 2]
        rho_R, u_R, P_R = out_R[:, 0], out_R[:, 1], out_R[:, 2]

        # Area at the (detached) interface location — reattach x_s for shock grad
        A_i = get_area(x_s).expand_as(rho_L)

        EPS = UnsteadyLoss.EPS
        H_L = GAMMA / (GAMMA - 1.0) * P_L / (rho_L + EPS)
        H_R = GAMMA / (GAMMA - 1.0) * P_R / (rho_R + EPS)

        flux_L_mass = rho_L * u_L * A_i
        flux_R_mass = rho_R * u_R * A_i
        flux_L_mom  = (rho_L * u_L ** 2 + P_L) * A_i
        flux_R_mom  = (rho_R * u_R ** 2 + P_R) * A_i
        flux_L_en   = rho_L * u_L * (H_L + 0.5 * u_L ** 2) * A_i
        flux_R_en   = rho_R * u_R * (H_R + 0.5 * u_R ** 2) * A_i

        rh_mass = (flux_L_mass - flux_R_mass).pow(2).mean()
        rh_mom  = (flux_L_mom  - flux_R_mom ).pow(2).mean()
        rh_en   = (flux_L_en   - flux_R_en  ).pow(2).mean()

        a_L = torch.sqrt(GAMMA * P_L / (rho_L + EPS) + EPS)
        a_R = torch.sqrt(GAMMA * P_R / (rho_R + EPS) + EPS)
        sup_pre  = torch.relu(a_L - u_L).pow(2).mean()   # penalise u_L < a_L
        sub_post = torch.relu(u_R - a_R).pow(2).mean()   # penalise u_R > a_R
        entropy  = torch.relu(P_L - P_R).pow(2).mean()   # penalise P drop the wrong way

        return rh_mass + rh_mom + rh_en + sup_pre + sub_post + entropy"""

for cell in nb["cells"]:
    src = get_src(cell)
    if "Rankine-Hugoniot conditions at x = x_s for each t in t_pts" in src:
        set_src(cell, src.replace(OLD_IFACE, NEW_IFACE))
        print("PATCH 3 applied: DualUnsteadyLoss.interface")
        break

# ──────────────────────────────────────────────────────────────────────────────
# PATCH 4  (cell-12  dual_network_training_loop)
# BUG:  retain_graph=True on the model backward leaves stale computation
#       graph nodes alive; when the shock backward runs on the same iface2
#       the graph has already been freed for some nodes → NaN/error.
# FIX:  compute interface loss once per step and split gradients cleanly:
#       model step uses loss.backward(retain_graph=True) but zeros shock grad.
#       shock step uses iface.backward() on the same retained graph (iface
#       is a sub-expression of loss so its graph is still live).
#       Also add NaN guard to break early with a diagnostic.
# ──────────────────────────────────────────────────────────────────────────────
OLD_LOOP = """\
        t_iface = _interface_times()

        # ── model step (both networks, frozen shock) ───────────────────────
        model_opt.zero_grad()

        pde_L    = DualUnsteadyLoss._pde_single(net_L, pts_L).pow(2).mean()
        pde_R    = DualUnsteadyLoss._pde_single(net_R, pts_R).pow(2).mean()
        bc_L     = DualUnsteadyLoss._left_bc(net_L, pts_L)
        bc_R     = DualUnsteadyLoss._right_bc(net_R, pts_R)
        iface    = DualUnsteadyLoss.interface(shock, net_L, net_R, t_iface)
        prior    = DUAL_NET_SHOCK_PRIOR_WEIGHT * (shock.clamped_x() - X_SHOCK_REF) ** 2

        loss = (
            pde_L + pde_R
            + DUAL_NET_BC_WEIGHT        * (bc_L + bc_R)
            + DUAL_NET_INTERFACE_WEIGHT * iface
            + prior
        )

        shock.x_shock.requires_grad_(False)
        loss.backward(retain_graph=True)
        shock.x_shock.requires_grad_(True)
        nn.utils.clip_grad_norm_(model_params, DUAL_NET_GRAD_CLIP)
        model_opt.step()

        # ── shock step (only interface loss) ──────────────────────────────
        shock_opt.zero_grad()
        iface2 = DualUnsteadyLoss.interface(shock, net_L, net_R, t_iface)
        (DUAL_NET_INTERFACE_WEIGHT * iface2 + prior).backward()
        shock_opt.step()"""

NEW_LOOP = """\
        t_iface = _interface_times()

        # ── compute all loss terms once ────────────────────────────────────
        pde_L = DualUnsteadyLoss._pde_single(net_L, pts_L).pow(2).mean()
        pde_R = DualUnsteadyLoss._pde_single(net_R, pts_R).pow(2).mean()
        bc_L  = DualUnsteadyLoss._left_bc(net_L,  pts_L)
        bc_R  = DualUnsteadyLoss._right_bc(net_R, pts_R)
        iface = DualUnsteadyLoss.interface(shock, net_L, net_R, t_iface)
        prior = DUAL_NET_SHOCK_PRIOR_WEIGHT * (shock.clamped_x() - X_SHOCK_REF) ** 2

        loss = (
            pde_L + pde_R
            + DUAL_NET_BC_WEIGHT        * (bc_L + bc_R)
            + DUAL_NET_INTERFACE_WEIGHT * iface
            + prior
        )

        # NaN guard — restore best and skip
        if not torch.isfinite(loss):
            if snap_L: net_L.load_state_dict(snap_L)
            if snap_R: net_R.load_state_dict(snap_R)
            if snap_I: shock.load_state_dict(snap_I)
            print(f"  NaN at epoch {epoch} — restored best snapshot, continuing")
            pts_L, pts_R = sampler_L(), sampler_R()
            continue

        # ── model step: zero shock grad so it does not move here ──────────
        model_opt.zero_grad()
        shock_opt.zero_grad()
        shock.x_shock.requires_grad_(False)
        loss.backward(retain_graph=True)   # keep graph for shock step below
        shock.x_shock.requires_grad_(True)
        nn.utils.clip_grad_norm_(model_params, DUAL_NET_GRAD_CLIP)
        model_opt.step()

        # ── shock step: only interface + prior, graph still live ──────────
        shock_opt.zero_grad()
        shock_loss = DUAL_NET_INTERFACE_WEIGHT * iface + prior
        shock_loss.backward()              # uses retained graph
        shock_opt.step()"""

for cell in nb["cells"]:
    src = get_src(cell)
    if (
        "── compute all loss terms once" in src
        or "── model step (both networks, frozen shock)" in src
    ):
        set_src(cell, src.replace(OLD_LOOP, NEW_LOOP))
        print("PATCH 4 applied: dual training loop")
        break

json.dump(nb, open(path, "w"), indent=1)
print("All patches written.")
