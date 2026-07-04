# Copyright 2025 Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""Classical optical performance evaluation for geometric lens systems. Accuracy aligned with Zemax.

Functions:
    Spot Diagram:
        - draw_spot_diagram(): Draw spot diagram with separate Field and Wavelength (Zemax-style grid layout)

    Distortion:
        - calc_distortion_2D(): Calculate distortion at a specific field angle
        - draw_distortion_radial(): Draw distortion curve vs field angle (Zemax format)
        - distortion_map(): Compute distortion map at a given depth
        - draw_distortion(): Draw distortion map visualization

    Chief Ray & Ray Aiming:
        - calc_chief_ray(): Compute chief ray for an incident angle
        - calc_chief_ray_infinite(): Compute chief ray for infinite object distance
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from ..basics import (
    DEFAULT_WAVE,
    DEPTH,
    EPSILON,
    GEO_GRID,
    SPP_CALC,
    SPP_PSF,
    WAVE_RGB,
)
from ..optics.ray import Ray

# RGB color definitions for wavelength visualization
RGB_RED = "#CC0000"
RGB_GREEN = "#006600"
RGB_BLUE = "#0066CC"
RGB_COLORS = [RGB_RED, RGB_GREEN, RGB_BLUE]
RGB_LABELS = ["R", "G", "B"]


class GeoLensEval:
    # ================================================================
    # Spot diagram
    # ================================================================
    @torch.no_grad()
    def draw_spot_diagram(
        self,
        save_name="./lens_spot_diagram.png",
        num_fov=5,
        depth=float("inf"),
        num_rays=SPP_PSF,
        wvln_list=WAVE_RGB,
        scale=200,
        show=False,
    ):
        """Draw spot diagram with separate Field and Wavelength (Zemax-style grid layout).

        This function creates a grid of spot diagrams: rows represent different field angles,
        columns represent different wavelengths.

        Args:
            save_name (string, optional): filename to save. Defaults to "./lens_spot_diagram.png".
            num_fov (int, optional): number of field angles. Defaults to 5.
            depth (float, optional): depth of the point source. Defaults to float("inf").
            num_rays (int, optional): number of rays to sample. Defaults to SPP_PSF.
            wvln_list (list, optional): wavelength list to render. Defaults to WAVE_RGB.
            scale (float, optional): scale factor for plot limits in micrometers. Defaults to 200.
            show (bool, optional): whether to show the plot. Defaults to False.

        Returns:
            tuple: (imax, imay) numpy arrays with shape (nWL, nField, nSample) in micrometers.
        """
        assert isinstance(wvln_list, list), "wvln_list must be a list"
        
        nWL = len(wvln_list)
        nField = num_fov
        nSample = num_rays

        # Initialize arrays to store spot positions: (nWL, nField, nSample)
        imax_list = []
        imay_list = []

        # Trace rays for each wavelength
        for wvln_idx, wvln in enumerate(wvln_list):
            # Sample rays along meridional (y) direction, shape [num_fov, num_rays, 3]
            ray = self.sample_radial_rays(
                num_field=num_fov, depth=depth, num_rays=num_rays, wvln=wvln
            )

            # Trace rays to sensor plane, shape [num_fov, num_rays, 3]
            ray = self.trace2sensor(ray)
            ray_o = ray.o.clone().cpu()  # shape [num_fov, num_rays, 3]
            ray_valid = ray.is_valid.clone().cpu()  # shape [num_fov, num_rays]

            # Extract x and y coordinates, filter valid rays
            # Shape: [num_fov, num_rays]
            x_data = ray_o[:, :, 0]  # x coordinates in mm
            y_data = ray_o[:, :, 1]  # y coordinates in mm

            # Store data: (nField, nSample) -> will be stacked to (nWL, nField, nSample)
            imax_list.append(x_data.numpy())
            imay_list.append(y_data.numpy())

        # Stack to get shape (nWL, nField, nSample)
        imax = np.stack(imax_list, axis=0)  # (nWL, nField, nSample)
        imay = np.stack(imay_list, axis=0)  # (nWL, nField, nSample)

        # Call the drawing function (converts mm to um internally and creates figure)
        imax_um, imay_um, fig = self._draw_spot_diagram_impl(imax, imay, scale=scale)

        # Save or show figure
        if show:
            plt.show()
        else:
            assert save_name.endswith(".png"), "save_name must end with .png"
            plt.savefig(save_name, bbox_inches="tight", format="png", dpi=300)
        plt.close(fig)

        return imax_um, imay_um

    def _draw_spot_diagram_impl(self, imax, imay, scale=200):
        """Internal implementation of spot diagram drawing.

        Args:
            imax (np.ndarray): x-coordinates of spot positions in mm, shape (nWL, nField, nSample)
            imay (np.ndarray): y-coordinates of spot positions in mm, shape (nWL, nField, nSample)
            scale (float): scale factor for plot limits in micrometers. Defaults to 200.

        Returns:
            tuple: (imax, imay) in micrometers, shape (nWL, nField, nSample)
        """
        # Convert from mm to micrometers
        imax_um = imax * 1000  # mm to um
        imay_um = imay * 1000  # mm to um

        nWL, nField, nSample = imax_um.shape

        # Color mapping for wavelengths
        colors = ['b', 'lime', 'r', 'y', 'g', 'c', 'm']  # Blue, Lime, Red, Yellow, Green, Cyan, Magenta

        # Create subplot grid: rows = fields, cols = wavelengths
        fig, axes = plt.subplots(nField, nWL, figsize=(nWL * 3, nField * 3))
        
        # Handle single row or single column case
        if nField == 1:
            axes = axes.reshape(1, -1)
        if nWL == 1:
            axes = axes.reshape(-1, 1)
        axes = np.atleast_2d(axes)

        # Draw each field and wavelength combination
        for i in range(nField):  # Field index (row)
            for j in range(nWL):  # Wavelength index (column)
                X = imax_um[j, i, :]
                Y = imay_um[j, i, :]
                spotPlot = axes[i, j]  # Select corresponding axis

                # Use color based on wavelength index
                color = colors[j % len(colors)]
                spotPlot.scatter(X, Y, c=color, marker='+', s=20)

                # Set equal aspect ratio
                spotPlot.set_aspect('equal', 'box')
                spotPlot.set_xlim(-scale * 1.1, scale * 1.1)
                spotPlot.set_ylim(-scale * 1.1, scale * 1.1)
                spotPlot.grid(color='gainsboro')

                # Set major and minor locators
                spotPlot.xaxis.set_major_locator(plt.MultipleLocator(scale / 1))
                spotPlot.xaxis.set_minor_locator(plt.MultipleLocator(scale / 5))
                spotPlot.yaxis.set_major_locator(plt.MultipleLocator(scale / 1))
                spotPlot.yaxis.set_minor_locator(plt.MultipleLocator(scale / 5))
                spotPlot.grid(True)

                # Remove tick labels for cleaner look
                spotPlot.set_xticklabels([])
                spotPlot.set_yticklabels([])

                # Show scale only on first column
                if j == 0:
                    axes[i, j].set_yticks([-scale, 0, scale])
                    axes[i, j].set_yticklabels([-scale, 0, scale], color='k', size=10)

        return imax_um, imay_um, fig

    # ================================================================
    # Distortion
    # ================================================================
    def calc_distortion_2D(
        self, rfov, wvln=DEFAULT_WAVE, plane="meridional", ray_aiming=True
    ):
        """Calculate distortion at a specific field angle.

        Args:
            rfov (float): view angle (degree)
            wvln (float): wavelength
            plane (str): meridional or sagittal
            ray_aiming (bool): whether the chief ray through the center of the stop.

        Returns:
            distortion (float): distortion at the specific field angle
        """
        # Calculate ideal image height
        eff_foclen = self.foclen
        ideal_imgh = eff_foclen * np.tan(rfov * np.pi / 180)

        # Calculate chief ray
        chief_ray_o, chief_ray_d = self.calc_chief_ray_infinite(
            rfov=rfov, wvln=wvln, plane=plane, ray_aiming=ray_aiming
        )
        ray = Ray(chief_ray_o, chief_ray_d, wvln=wvln, device=self.device)

        ray, _ = self.trace(ray)
        t = (self.d_sensor - ray.o[..., 2]) / ray.d[..., 2]

        # Calculate actual image height
        if plane == "sagittal":
            actual_imgh = (ray.o[..., 0] + ray.d[..., 0] * t).abs()
        elif plane == "meridional":
            actual_imgh = (ray.o[..., 1] + ray.d[..., 1] * t).abs()
        else:
            raise ValueError(f"Invalid plane: {plane}")

        # Calculate distortion
        actual_imgh = actual_imgh.cpu().numpy()
        ideal_imgh = ideal_imgh.cpu().numpy()
        distortion = (actual_imgh - ideal_imgh) / ideal_imgh

        # Handle the case where ideal_imgh is 0 or very close to 0
        mask = abs(ideal_imgh) < EPSILON
        distortion[mask] = 0.0

        return distortion

    def draw_distortion_radial(
        self,
        rfov,
        save_name=None,
        num_points=GEO_GRID,
        wvln=DEFAULT_WAVE,
        plane="meridional",
        ray_aiming=True,
        show=False,
    ):
        """Draw distortion. zemax format(default): ray_aiming = False.

        Note: this function is provided by a community contributor.

        Args:
            rfov: view angle (degrees)
            save_name: Save filename. Defaults to None.
            num_points: Number of points. Defaults to GEO_GRID.
            plane: Meridional or sagittal. Defaults to meridional.
            ray_aiming: Whether to use ray aiming. Defaults to False.
        """
        # Sample view angles
        rfov_samples = torch.linspace(0, rfov, num_points)
        distortions = []

        # Calculate distortion
        distortions = self.calc_distortion_2D(
            rfov=rfov_samples,
            wvln=wvln,
            plane=plane,
            ray_aiming=ray_aiming,
        )

        # Handle possible NaN values and convert to percentage
        values = [
            t.item() * 100 if not math.isnan(t.item()) else 0 for t in distortions
        ]

        # Create figure (taller aspect ratio for a clearer field axis)
        fig, ax = plt.subplots(figsize=(8, 12))
        ax.set_title(f"{plane} Surface Distortion", fontsize=20, fontweight="bold")

        # Draw distortion curve
        ax.plot(values, rfov_samples, linestyle="-", color="g", linewidth=2.2)

        # Draw reference line (vertical line)
        ax.axvline(x=0, color="k", linestyle="-", linewidth=0.8)

        # Set grid
        ax.grid(True, color="gray", linestyle="-", linewidth=0.5, alpha=1)

        # Dynamically adjust x-axis range
        value = max(abs(v) for v in values)
        margin = value * 0.2  # 20% margin
        x_min, x_max = -max(0.2, value + margin), max(0.2, value + margin)

        # Set ticks
        x_ticks = np.linspace(-value, value, 3)
        y_ticks = np.linspace(0, rfov, 3)

        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)

        # Format tick labels
        x_labels = [f"{x:.1f}" for x in x_ticks]
        y_labels = [f"{y:.1f}" for y in y_ticks]

        ax.set_xticklabels(x_labels)
        ax.set_yticklabels(y_labels)
        ax.tick_params(axis="both", labelsize=16)

        # Set axis labels
        ax.set_xlabel("Distortion (%)", fontsize=18)
        ax.set_ylabel("Field of View (degrees)", fontsize=18)

        # Set axis range
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, rfov)

        if show:
            plt.show()
        else:
            if save_name is None:
                save_name = f"./{plane}_distortion_inf.png"
            plt.savefig(save_name, bbox_inches="tight", format="png", dpi=300)
        plt.close(fig)

    @torch.no_grad()
    def distortion_map(self, num_grid=16, depth=DEPTH, wvln=DEFAULT_WAVE):
        """Compute distortion map at a given depth.

        Args:
            num_grid (int or tuple): number of grid points. If int, creates square grid (num_grid, num_grid).
            depth (float): depth of the point source.
            wvln (float): wavelength.

        Returns:
            distortion_grid (torch.Tensor): distortion map. shape (grid_size, grid_size, 2)
        """
        # Convert int to tuple if needed
        if isinstance(num_grid, int):
            num_grid = (num_grid, num_grid)
        
        # Sample and trace rays, shape (grid_size, grid_size, num_rays, 3)
        ray = self.sample_grid_rays(depth=depth, num_grid=num_grid, wvln=wvln, uniform_fov=False)
        ray = self.trace2sensor(ray)

        # Calculate centroid of the rays, shape (grid_size, grid_size, 2)
        ray_xy = ray.centroid()[..., :2]
        x_dist = -ray_xy[..., 0] / self.sensor_size[1] * 2
        y_dist = ray_xy[..., 1] / self.sensor_size[0] * 2
        distortion_grid = torch.stack((x_dist, y_dist), dim=-1)
        return distortion_grid

    def draw_distortion(
        self, save_name=None, num_grid=16, depth=DEPTH, wvln=DEFAULT_WAVE, show=False
    ):
        """Draw distortion map.

        Args:
            save_name (str, optional): filename to save. Defaults to None.
            num_grid (int, optional): number of grid points. Defaults to 16.
            depth (float, optional): depth of the point source. Defaults to DEPTH.
            wvln (float, optional): wavelength. Defaults to DEFAULT_WAVE.
            show (bool, optional): whether to show the plot. Defaults to False.
        """
        # Ray tracing to calculate distortion map
        distortion_grid = self.distortion_map(num_grid=num_grid, depth=depth, wvln=wvln)
        x1 = distortion_grid[..., 0].cpu().numpy()
        y1 = distortion_grid[..., 1].cpu().numpy()

        # Draw image
        fig, ax = plt.subplots()
        ax.set_title("Lens distortion")
        ax.scatter(x1, y1, s=2)
        ax.axis("scaled")
        ax.grid(True)

        # Add grid lines based on grid_size
        ax.set_xticks(np.linspace(-1, 1, num_grid))
        ax.set_yticks(np.linspace(-1, 1, num_grid))

        if show:
            plt.show()
        else:
            depth_str = "inf" if depth == float("inf") else f"{-depth}mm"
            if save_name is None:
                save_name = f"./distortion_{depth_str}.png"
            plt.savefig(save_name, bbox_inches="tight", format="png", dpi=300)
        plt.close(fig)

    # ================================================================
    # Chief ray calculation and ray aiming
    # ================================================================
    @torch.no_grad()
    def calc_chief_ray(self, fov, plane="sagittal"):
        """Compute chief ray for an incident angle.

        If chief ray is only used to determine the ideal image height, we can warp this function into the image height calculation function.

        Args:
            fov (float): incident angle in degree.
            plane (str): "sagittal" or "meridional".

        Returns:
            chief_ray_o (torch.Tensor): origin of chief ray.
            chief_ray_d (torch.Tensor): direction of chief ray.

        Note:
            It is 2D ray tracing, for 3D chief ray, we can shrink the pupil, trace rays, calculate the centroid as the chief ray.
        """
        # Sample parallel rays from object space
        ray = self.sample_parallel_2D(
            fov=fov, num_rays=SPP_CALC, entrance_pupil=True, plane=plane
        )
        inc_ray = ray.clone()

        # Trace to the aperture
        surf_range = range(0, self.aper_idx)
        ray, _ = self.trace(ray, surf_range=surf_range)

        # Look for the ray that is closest to the optical axis
        center_x = torch.min(torch.abs(ray.o[:, 0]))
        center_idx = torch.where(torch.abs(ray.o[:, 0]) == center_x)[0][0].item()
        chief_ray_o, chief_ray_d = inc_ray.o[center_idx, :], inc_ray.d[center_idx, :]

        return chief_ray_o, chief_ray_d

    @torch.no_grad()
    def calc_chief_ray_infinite(
        self,
        rfov,
        depth=0.0,
        wvln=DEFAULT_WAVE,
        plane="meridional",
        num_rays=SPP_CALC,
        ray_aiming=True,
    ):
        """Compute chief ray for an incident angle.

        Args:
            rfov (float): incident angle in degree.
            depth (float): depth of the object.
            wvln (float): wavelength of the light.
            plane (str): "sagittal" or "meridional".
            num_rays (int): number of rays.
            ray_aiming (bool): whether the chief ray through the center of the stop.
        """
        if isinstance(rfov, float) and rfov > 0:
            rfov = torch.linspace(0, rfov, 2)
        rfov = rfov.to(self.device)

        if not isinstance(depth, torch.Tensor):
            depth = torch.tensor(depth, device=self.device).repeat(len(rfov))

        # set chief ray
        chief_ray_o = torch.zeros([len(rfov), 3]).to(self.device)
        chief_ray_d = torch.zeros([len(rfov), 3]).to(self.device)

        # Convert rfov to radian
        rfov = rfov * torch.pi / 180.0

        if torch.any(rfov == 0):
            chief_ray_o[0, ...] = torch.tensor(
                [0.0, 0.0, depth[0]], device=self.device, dtype=torch.float32
            )
            chief_ray_d[0, ...] = torch.tensor(
                [0.0, 0.0, 1.0], device=self.device, dtype=torch.float32
            )
            if len(rfov) == 1:
                return chief_ray_o, chief_ray_d

        if len(rfov) > 1:
            rfovs = rfov[1:]
            depths = depth[1:]

        if self.aper_idx == 0:
            if plane == "sagittal":
                chief_ray_o[1:, ...] = torch.stack(
                    [depths * torch.tan(rfovs), torch.zeros_like(rfovs), depths], dim=-1
                )
                chief_ray_d[1:, ...] = torch.stack(
                    [torch.sin(rfovs), torch.zeros_like(rfovs), torch.cos(rfovs)],
                    dim=-1,
                )
            else:
                chief_ray_o[1:, ...] = torch.stack(
                    [torch.zeros_like(rfovs), depths * torch.tan(rfovs), depths], dim=-1
                )
                chief_ray_d[1:, ...] = torch.stack(
                    [torch.zeros_like(rfovs), torch.sin(rfovs), torch.cos(rfovs)],
                    dim=-1,
                )

            return chief_ray_o, chief_ray_d

        # Scale factor
        pupilz, _ = self.calc_entrance_pupil()
        y_distance = torch.tan(rfovs) * (abs(depths) + pupilz)

        if ray_aiming:
            scale = 0.05
            delta = scale * y_distance

        if not ray_aiming:
            if plane == "sagittal":
                chief_ray_o[1:, ...] = torch.stack(
                    [-y_distance, torch.zeros_like(rfovs), depths], dim=-1
                )
                chief_ray_d[1:, ...] = torch.stack(
                    [torch.sin(rfovs), torch.zeros_like(rfovs), torch.cos(rfovs)],
                    dim=-1,
                )
            else:
                chief_ray_o[1:, ...] = torch.stack(
                    [torch.zeros_like(rfovs), -y_distance, depths], dim=-1
                )
                chief_ray_d[1:, ...] = torch.stack(
                    [torch.zeros_like(rfovs), torch.sin(rfovs), torch.cos(rfovs)],
                    dim=-1,
                )

        else:
            min_y = -y_distance - delta
            max_y = -y_distance + delta
            o1_linspace = torch.stack(
                [
                    torch.linspace(min_y[i], max_y[i], num_rays)
                    for i in range(len(min_y))
                ],
                dim=0,
            )

            o1 = torch.zeros([len(rfovs), num_rays, 3])
            o1[:, :, 2] = depths[0]

            o2_linspace = torch.stack(
                [
                    torch.linspace(-delta[i], delta[i], num_rays)
                    for i in range(len(min_y))
                ],
                dim=0,
            )

            o2 = torch.zeros([len(rfovs), num_rays, 3])
            o2[:, :, 2] = pupilz

            if plane == "sagittal":
                o1[:, :, 0] = o1_linspace
                o2[:, :, 0] = o2_linspace
            else:
                o1[:, :, 1] = o1_linspace
                o2[:, :, 1] = o2_linspace

            # Trace until the aperture
            ray = Ray(o1, o2 - o1, wvln=wvln, device=self.device)
            inc_ray = ray.clone()
            surf_range = range(0, self.aper_idx + 1)
            ray, _ = self.trace(ray, surf_range=surf_range)

            # Look for the ray that is closest to the optical axis
            if plane == "sagittal":
                _, center_idx = torch.min(torch.abs(ray.o[..., 0]), dim=1)
                chief_ray_o[1:, ...] = inc_ray.o[
                    torch.arange(len(rfovs)), center_idx.long(), ...
                ]
                chief_ray_d[1:, ...] = torch.stack(
                    [torch.sin(rfovs), torch.zeros_like(rfovs), torch.cos(rfovs)],
                    dim=-1,
                )
            else:
                _, center_idx = torch.min(torch.abs(ray.o[..., 1]), dim=1)
                chief_ray_o[1:, ...] = inc_ray.o[
                    torch.arange(len(rfovs)), center_idx.long(), ...
                ]
                chief_ray_d[1:, ...] = torch.stack(
                    [torch.zeros_like(rfovs), torch.sin(rfovs), torch.cos(rfovs)],
                    dim=-1,
                )

        return chief_ray_o, chief_ray_d
