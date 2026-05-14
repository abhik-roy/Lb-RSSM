import numpy as np


class Quadrotor:
    m    = 0.5
    g    = 9.81
    Ixx  = 4.9e-3
    Iyy  = 4.9e-3
    Izz  = 8.8e-3

    def __init__(self, dt=0.001, wind_scale=1.0):
        self.dt         = dt
        self.wind_scale = wind_scale

    @staticmethod
    def rotation_matrix(phi, theta, psi):
        cphi, sphi = np.cos(phi), np.sin(phi)
        cth,  sth  = np.cos(theta), np.sin(theta)
        cpsi, spsi = np.cos(psi), np.sin(psi)
        return np.array([
            [cth*cpsi, sphi*sth*cpsi - cphi*spsi, cphi*sth*cpsi + sphi*spsi],
            [cth*spsi, sphi*sth*spsi + cphi*cpsi, cphi*sth*spsi - sphi*cpsi],
            [-sth,     sphi*cth,                  cphi*cth                 ],
        ])

    @staticmethod
    def W_matrix(phi, theta):
        cphi, sphi = np.cos(phi), np.sin(phi)
        cth = np.cos(theta)
        if abs(cth) < 1e-6:
            cth = 1e-6
        return np.array([
            [1, sphi*np.tan(theta), cphi*np.tan(theta)],
            [0, cphi,               -sphi             ],
            [0, sphi/cth,            cphi/cth          ],
        ])

    def wind(self, t, pos):
        # Purely temporal disturbance: no spatial component.
        # φ(pos, vel) carries zero information about wind phase, so W_hat
        # converges to the time-average (≈ 0) and the residual error persists.
        # RSSM: h_t tracks the temporal cycle implicitly → error drops.
        ws = self.wind_scale
        Fx = ws * 0.8*np.sin(0.4*t)
        Fy = ws * 0.6*np.cos(0.3*t)
        Fz = ws * 0.35*np.sin(0.5*t)
        return np.array([Fx, Fy, Fz])

    def derivatives(self, state, T, tau, t):
        x, y, z, vx, vy, vz, phi, theta, psi, p, q, r = state
        pos = np.array([x, y, z])

        R      = self.rotation_matrix(phi, theta, psi)
        W      = self.W_matrix(phi, theta)
        F_dist = self.wind(t, pos)

        thrust_world = R @ np.array([0.0, 0.0, float(T)])
        a_xyz  = (thrust_world + F_dist) / self.m - np.array([0.0, 0.0, self.g])

        omega  = np.array([p, q, r])
        I_diag = np.array([self.Ixx, self.Iyy, self.Izz])
        gyro   = np.cross(omega, I_diag * omega)
        alpha  = (np.asarray(tau, dtype=float) - gyro) / I_diag

        eta_dot = W @ omega

        return np.array([vx, vy, vz,
                         a_xyz[0], a_xyz[1], a_xyz[2],
                         eta_dot[0], eta_dot[1], eta_dot[2],
                         alpha[0], alpha[1], alpha[2]])

    def step(self, state, T, tau, t):
        dt = self.dt
        k1 = self.derivatives(state,           T, tau, t)
        k2 = self.derivatives(state + dt/2*k1, T, tau, t + dt/2)
        k3 = self.derivatives(state + dt/2*k2, T, tau, t + dt/2)
        k4 = self.derivatives(state + dt*k3,   T, tau, t + dt)
        new_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        new_state[6:9] = ((new_state[6:9] + np.pi) % (2*np.pi)) - np.pi
        return new_state

    @staticmethod
    def figure8(t, scale=1.0, height=1.5, omega=0.4):
        d  = 2.0 / (3 - np.cos(2*omega*t))
        xd = scale * d * np.cos(omega*t)
        yd = scale * d * np.sin(2*omega*t) / 2
        zd = height
        eps = 1e-4
        d2  = 2.0 / (3 - np.cos(2*omega*(t+eps)))
        xd2 = scale * d2 * np.cos(omega*(t+eps))
        yd2 = scale * d2 * np.sin(2*omega*(t+eps)) / 2
        vxd = (xd2 - xd) / eps
        vyd = (yd2 - yd) / eps
        return np.array([xd, yd, zd]), np.array([vxd, vyd, 0.0]), np.zeros(3)
