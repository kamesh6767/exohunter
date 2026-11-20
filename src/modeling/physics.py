"""Astrophysical parameter extraction from transit signals and host star properties."""
from dataclasses import dataclass
import numpy as np
from ..detection.bls import TransitCandidate
from ..data.loader import LightCurveData


@dataclass
class PhysicalProperties:
    """Derived physical and orbital characteristics of an exoplanet candidate."""
    target_name: str
    radius_earth: float           # Planetary radius in Earth radii (R_earth)
    radius_jupiter: float         # Planetary radius in Jupiter radii (R_jup)
    semi_major_axis_au: float     # Orbital semi-major axis in Astronomical Units (AU)
    orbital_period_days: float    # Orbital period in days
    equilibrium_temp_k: float     # Estimated planetary equilibrium temperature (Kelvin)
    planet_type: str              # e.g., 'Terrestrial', 'Super-Earth', 'Sub-Neptune', 'Gas Giant'
    habitable_zone_status: str    # 'In Habitable Zone', 'Too Hot', 'Too Cold'
    transit_depth_ppm: float      # Transit depth in parts per million (ppm)


class ExoplanetPhysicsModel:
    """Calculates physical exoplanet dimensions using Kepler's Laws and transit geometry."""

    SOLAR_TO_EARTH_RADIUS = 109.209  # 1 R_sun = 109.209 R_earth
    JUPITER_TO_EARTH_RADIUS = 11.209 # 1 R_jup = 11.209 R_earth
    AU_TO_SOLAR_RADIUS = 215.032     # 1 AU = 215.032 R_sun

    def compute(self, candidate: TransitCandidate, stellar_data: LightCurveData) -> PhysicalProperties:
        """Derive physical parameters from transit candidate and stellar properties."""
        r_star = max(stellar_data.stellar_radius, 0.05)
        m_star = max(stellar_data.stellar_mass, 0.05)
        teff = max(stellar_data.stellar_teff, 1500.0)

        # 1. Planetary Radius: delta = (R_p / R_*)^2
        depth = max(candidate.depth, 1e-7)
        radius_ratio = np.sqrt(depth)
        radius_earth = float(radius_ratio * r_star * self.SOLAR_TO_EARTH_RADIUS)
        radius_jupiter = float(radius_earth / self.JUPITER_TO_EARTH_RADIUS)

        # 2. Semi-Major Axis via Kepler's Third Law:
        # a [AU] = (M_* [M_sun] * (P [days] / 365.25)^2 )^(1/3)
        period_years = candidate.period / 365.25
        semi_major_axis_au = float((m_star * (period_years ** 2)) ** (1.0 / 3.0))

        # 3. Planetary Equilibrium Temperature (Assuming Bond Albedo A=0.3, uniform redistribution):
        # T_eq = T_eff * sqrt(R_* / (2 * a))
        # Note: a in R_sun units = a_AU * 215.032
        a_in_rsun = semi_major_axis_au * self.AU_TO_SOLAR_RADIUS
        teq = float(teff * np.sqrt(r_star / (2.0 * a_in_rsun)) * ((1.0 - 0.3) ** 0.25))

        # 4. Planetary Classification by Radius
        if radius_earth < 1.25:
            planet_type = "Earth-sized / Terrestrial"
        elif radius_earth < 2.0:
            planet_type = "Super-Earth"
        elif radius_earth < 4.0:
            planet_type = "Sub-Neptune"
        elif radius_earth < 15.0:
            planet_type = "Gas Giant (Jovian)"
        else:
            planet_type = "Stellar / Brown Dwarf Candidate"

        # 5. Habitable Zone Status (Conservative equilibrium range: ~180 K to 310 K)
        if 180.0 <= teq <= 315.0:
            habitable_zone_status = "In Habitable Zone (Liquid water possible)"
        elif teq > 315.0:
            habitable_zone_status = f"Too Hot ({teq:.0f} K - Exceeds boiling limit)"
        else:
            habitable_zone_status = f"Too Cold ({teq:.0f} K - Below freezing threshold)"

        return PhysicalProperties(
            target_name=candidate.target_name,
            radius_earth=radius_earth,
            radius_jupiter=radius_jupiter,
            semi_major_axis_au=semi_major_axis_au,
            orbital_period_days=candidate.period,
            equilibrium_temp_k=teq,
            planet_type=planet_type,
            habitable_zone_status=habitable_zone_status,
            transit_depth_ppm=depth * 1e6,
        )
