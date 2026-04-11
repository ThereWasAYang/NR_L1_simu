from __future__ import annotations

from dataclasses import dataclass


RAY_OFFSETS = (
    0.0447,
    -0.0447,
    0.1413,
    -0.1413,
    0.2492,
    -0.2492,
    0.3715,
    -0.3715,
    0.5129,
    -0.5129,
    0.6797,
    -0.6797,
    0.8844,
    -0.8844,
    1.1481,
    -1.1481,
    1.5195,
    -1.5195,
    2.1551,
    -2.1551,
)


@dataclass(frozen=True)
class TdlTap:
    normalized_delay: float
    power_db: float
    fading: str


@dataclass(frozen=True)
class CdlCluster:
    normalized_delay: float
    power_db: float
    aod_deg: float
    aoa_deg: float
    zod_deg: float
    zoa_deg: float
    fading: str = "Rayleigh"


@dataclass(frozen=True)
class CdlProfile:
    clusters: tuple[CdlCluster, ...]
    c_asd_deg: float
    c_asa_deg: float
    c_zsd_deg: float
    c_zsa_deg: float
    xpr_db: float


TDL_PROFILES: dict[str, tuple[TdlTap, ...]] = {
    "TDL-A": (
        TdlTap(0.0000, -13.4, "Rayleigh"),
        TdlTap(0.3819, 0.0, "Rayleigh"),
        TdlTap(0.4025, -2.2, "Rayleigh"),
        TdlTap(0.5868, -4.0, "Rayleigh"),
        TdlTap(0.4610, -6.0, "Rayleigh"),
        TdlTap(0.5375, -8.2, "Rayleigh"),
        TdlTap(0.6708, -9.9, "Rayleigh"),
        TdlTap(0.5750, -10.5, "Rayleigh"),
        TdlTap(0.7618, -7.5, "Rayleigh"),
        TdlTap(1.5375, -15.9, "Rayleigh"),
        TdlTap(1.8978, -6.6, "Rayleigh"),
        TdlTap(2.2242, -16.7, "Rayleigh"),
        TdlTap(2.1718, -12.4, "Rayleigh"),
        TdlTap(2.4942, -15.2, "Rayleigh"),
        TdlTap(2.5119, -10.8, "Rayleigh"),
        TdlTap(3.0582, -11.3, "Rayleigh"),
        TdlTap(4.0810, -12.7, "Rayleigh"),
        TdlTap(4.4579, -16.2, "Rayleigh"),
        TdlTap(4.5695, -18.3, "Rayleigh"),
        TdlTap(4.7966, -18.9, "Rayleigh"),
        TdlTap(5.0066, -16.6, "Rayleigh"),
        TdlTap(5.3043, -19.9, "Rayleigh"),
        TdlTap(9.6586, -29.7, "Rayleigh"),
    ),
    "TDL-B": (
        TdlTap(0.0000, 0.0, "Rayleigh"),
        TdlTap(0.1072, -2.2, "Rayleigh"),
        TdlTap(0.2155, -4.0, "Rayleigh"),
        TdlTap(0.2095, -3.2, "Rayleigh"),
        TdlTap(0.2870, -9.8, "Rayleigh"),
        TdlTap(0.2986, -1.2, "Rayleigh"),
        TdlTap(0.3752, -3.4, "Rayleigh"),
        TdlTap(0.5055, -5.2, "Rayleigh"),
        TdlTap(0.3681, -7.6, "Rayleigh"),
        TdlTap(0.3697, -3.0, "Rayleigh"),
        TdlTap(0.5700, -8.9, "Rayleigh"),
        TdlTap(0.5283, -9.0, "Rayleigh"),
        TdlTap(1.1021, -4.8, "Rayleigh"),
        TdlTap(1.2756, -5.7, "Rayleigh"),
        TdlTap(1.5474, -7.5, "Rayleigh"),
        TdlTap(1.7842, -1.9, "Rayleigh"),
        TdlTap(2.0169, -7.6, "Rayleigh"),
        TdlTap(2.8294, -12.2, "Rayleigh"),
        TdlTap(3.0219, -9.8, "Rayleigh"),
        TdlTap(3.6187, -11.4, "Rayleigh"),
        TdlTap(4.1067, -14.9, "Rayleigh"),
        TdlTap(4.2790, -9.2, "Rayleigh"),
        TdlTap(4.7834, -11.3, "Rayleigh"),
    ),
    "TDL-C": (
        TdlTap(0.0000, -4.4, "Rayleigh"),
        TdlTap(0.2099, -1.2, "Rayleigh"),
        TdlTap(0.2219, -3.5, "Rayleigh"),
        TdlTap(0.2329, -5.2, "Rayleigh"),
        TdlTap(0.2176, -2.5, "Rayleigh"),
        TdlTap(0.6366, 0.0, "Rayleigh"),
        TdlTap(0.6448, -2.2, "Rayleigh"),
        TdlTap(0.6560, -3.9, "Rayleigh"),
        TdlTap(0.6584, -7.4, "Rayleigh"),
        TdlTap(0.7935, -7.1, "Rayleigh"),
        TdlTap(0.8213, -10.7, "Rayleigh"),
        TdlTap(0.9336, -11.1, "Rayleigh"),
        TdlTap(1.2285, -5.1, "Rayleigh"),
        TdlTap(1.3083, -6.8, "Rayleigh"),
        TdlTap(2.1704, -8.7, "Rayleigh"),
        TdlTap(2.7105, -13.2, "Rayleigh"),
        TdlTap(4.2589, -13.9, "Rayleigh"),
        TdlTap(4.6003, -13.9, "Rayleigh"),
        TdlTap(5.4902, -15.8, "Rayleigh"),
        TdlTap(5.6077, -17.1, "Rayleigh"),
        TdlTap(6.3065, -16.0, "Rayleigh"),
        TdlTap(6.6374, -15.7, "Rayleigh"),
        TdlTap(7.0427, -21.6, "Rayleigh"),
        TdlTap(8.6523, -22.8, "Rayleigh"),
    ),
    "TDL-D": (
        TdlTap(0.0000, -0.2, "LOS"),
        TdlTap(0.0000, -13.5, "Rayleigh"),
        TdlTap(0.0350, -18.8, "Rayleigh"),
        TdlTap(0.6120, -21.0, "Rayleigh"),
        TdlTap(1.3630, -22.8, "Rayleigh"),
        TdlTap(1.4050, -17.9, "Rayleigh"),
        TdlTap(1.8040, -20.1, "Rayleigh"),
        TdlTap(2.5960, -21.9, "Rayleigh"),
        TdlTap(1.7750, -22.9, "Rayleigh"),
        TdlTap(4.0420, -27.8, "Rayleigh"),
        TdlTap(7.9370, -23.6, "Rayleigh"),
        TdlTap(9.4240, -24.8, "Rayleigh"),
        TdlTap(9.7080, -30.0, "Rayleigh"),
        TdlTap(12.5250, -27.7, "Rayleigh"),
    ),
    "TDL-E": (
        TdlTap(0.0000, -0.03, "LOS"),
        TdlTap(0.0000, -22.03, "Rayleigh"),
        TdlTap(0.5133, -15.8, "Rayleigh"),
        TdlTap(0.5440, -18.1, "Rayleigh"),
        TdlTap(0.5630, -19.8, "Rayleigh"),
        TdlTap(0.5440, -22.9, "Rayleigh"),
        TdlTap(0.7112, -22.4, "Rayleigh"),
        TdlTap(1.9092, -18.6, "Rayleigh"),
        TdlTap(1.9293, -20.8, "Rayleigh"),
        TdlTap(1.9589, -22.6, "Rayleigh"),
        TdlTap(2.6426, -22.3, "Rayleigh"),
        TdlTap(3.7136, -25.6, "Rayleigh"),
        TdlTap(5.4524, -20.2, "Rayleigh"),
        TdlTap(12.0034, -29.8, "Rayleigh"),
        TdlTap(20.6519, -29.2, "Rayleigh"),
    ),
}


TDL_LOS_K_DB = {"TDL-D": 13.3, "TDL-E": 22.0}


CDL_PROFILES: dict[str, CdlProfile] = {
    "CDL-A": CdlProfile(
        clusters=(
            CdlCluster(0.0000, -13.4, -178.1, 51.3, 50.2, 125.4),
            CdlCluster(0.3819, 0.0, -4.2, -152.7, 93.2, 91.3),
            CdlCluster(0.4025, -2.2, -4.2, -152.7, 93.2, 91.3),
            CdlCluster(0.5868, -4.0, -4.2, -152.7, 93.2, 91.3),
            CdlCluster(0.4610, -6.0, 90.2, 76.6, 122.0, 94.0),
            CdlCluster(0.5375, -8.2, 90.2, 76.6, 122.0, 94.0),
            CdlCluster(0.6708, -9.9, 90.2, 76.6, 122.0, 94.0),
            CdlCluster(0.5750, -10.5, 121.5, -1.8, 150.2, 47.1),
            CdlCluster(0.7618, -7.5, -81.7, -41.9, 55.2, 56.0),
            CdlCluster(1.5375, -15.9, 158.4, 94.2, 26.4, 30.1),
            CdlCluster(1.8978, -6.6, -83.0, 51.9, 126.4, 58.8),
            CdlCluster(2.2242, -16.7, 134.8, -115.9, 171.6, 26.0),
            CdlCluster(2.1718, -12.4, -153.0, 26.6, 151.4, 49.2),
            CdlCluster(2.4942, -15.2, -172.0, 76.6, 157.2, 143.1),
            CdlCluster(2.5119, -10.8, -129.9, -7.0, 47.2, 117.4),
            CdlCluster(3.0582, -11.3, -136.0, -23.0, 40.4, 122.7),
            CdlCluster(4.0810, -12.7, 165.4, -47.2, 43.3, 123.2),
            CdlCluster(4.4579, -16.2, 148.4, 110.4, 161.8, 32.6),
            CdlCluster(4.5695, -18.3, 132.7, 144.5, 10.8, 27.2),
            CdlCluster(4.7966, -18.9, -118.6, 155.3, 16.7, 15.2),
            CdlCluster(5.0066, -16.6, -154.1, 102.0, 171.7, 146.0),
            CdlCluster(5.3043, -19.9, 126.5, -151.8, 22.7, 150.7),
            CdlCluster(9.6586, -29.7, -56.2, 55.2, 144.9, 156.1),
        ),
        c_asd_deg=5.0,
        c_asa_deg=11.0,
        c_zsd_deg=3.0,
        c_zsa_deg=3.0,
        xpr_db=10.0,
    ),
    "CDL-B": CdlProfile(
        clusters=(
            CdlCluster(0.0000, 0.0, 9.3, -173.3, 105.8, 78.9),
            CdlCluster(0.1072, -2.2, 9.3, -173.3, 105.8, 78.9),
            CdlCluster(0.2155, -4.0, 9.3, -173.3, 105.8, 78.9),
            CdlCluster(0.2095, -3.2, -34.1, 125.5, 115.3, 63.3),
            CdlCluster(0.2870, -9.8, -65.4, -88.0, 119.3, 59.9),
            CdlCluster(0.2986, -1.2, -11.4, 155.1, 103.2, 67.5),
            CdlCluster(0.3752, -3.4, -11.4, 155.1, 103.2, 67.5),
            CdlCluster(0.5055, -5.2, -11.4, 155.1, 103.2, 67.5),
            CdlCluster(0.3681, -7.6, -67.2, -89.8, 118.2, 82.6),
            CdlCluster(0.3697, -3.0, 52.5, 132.1, 102.0, 66.3),
            CdlCluster(0.5700, -8.9, -72.0, -83.6, 100.4, 61.6),
            CdlCluster(0.5283, -9.0, 74.3, 95.3, 98.3, 58.0),
            CdlCluster(1.1021, -4.8, -52.2, 103.7, 103.4, 78.2),
            CdlCluster(1.2756, -5.7, -50.5, -87.8, 102.5, 82.0),
            CdlCluster(1.5474, -7.5, 61.4, -92.5, 101.4, 62.4),
            CdlCluster(1.7842, -1.9, 30.6, -139.1, 103.0, 78.0),
            CdlCluster(2.0169, -7.6, -72.5, -90.6, 100.0, 60.9),
            CdlCluster(2.8294, -12.2, -90.6, 58.6, 115.2, 82.9),
            CdlCluster(3.0219, -9.8, -77.6, -79.0, 100.5, 60.8),
            CdlCluster(3.6187, -11.4, -82.6, 65.8, 119.6, 57.3),
            CdlCluster(4.1067, -14.9, -103.6, 52.7, 118.7, 59.9),
            CdlCluster(4.2790, -9.2, 75.6, 88.7, 117.8, 60.1),
            CdlCluster(4.7834, -11.3, -77.6, -60.4, 115.7, 62.3),
        ),
        c_asd_deg=10.0,
        c_asa_deg=22.0,
        c_zsd_deg=3.0,
        c_zsa_deg=7.0,
        xpr_db=8.0,
    ),
    "CDL-C": CdlProfile(
        clusters=(
            CdlCluster(0.0000, -4.4, -46.6, -101.0, 97.2, 87.6),
            CdlCluster(0.2099, -1.2, -22.8, 120.0, 98.6, 72.1),
            CdlCluster(0.2219, -3.5, -22.8, 120.0, 98.6, 72.1),
            CdlCluster(0.2329, -5.2, -22.8, 120.0, 98.6, 72.1),
            CdlCluster(0.2176, -2.5, -40.7, -127.5, 100.6, 70.1),
            CdlCluster(0.6366, 0.0, 0.3, 170.4, 99.2, 75.3),
            CdlCluster(0.6448, -2.2, 0.3, 170.4, 99.2, 75.3),
            CdlCluster(0.6560, -3.9, 0.3, 170.4, 99.2, 75.3),
            CdlCluster(0.6584, -7.4, 73.1, 55.4, 105.2, 67.4),
            CdlCluster(0.7935, -7.1, -64.5, 66.5, 95.3, 63.8),
            CdlCluster(0.8213, -10.7, 80.2, -48.1, 106.1, 71.4),
            CdlCluster(0.9336, -11.1, -97.1, 46.9, 93.5, 60.5),
            CdlCluster(1.2285, -5.1, -55.3, 68.1, 103.7, 90.6),
            CdlCluster(1.3083, -6.8, -64.3, -68.7, 104.2, 60.1),
            CdlCluster(2.1704, -8.7, -78.5, 81.5, 93.0, 61.0),
            CdlCluster(2.7105, -13.2, 102.7, 30.7, 104.2, 100.7),
            CdlCluster(4.2589, -13.9, 99.2, -16.4, 94.9, 62.3),
            CdlCluster(4.6003, -13.9, 88.8, 3.8, 93.1, 66.7),
            CdlCluster(5.4902, -15.8, -101.9, -13.7, 92.2, 52.9),
            CdlCluster(5.6077, -17.1, 92.2, 9.7, 106.7, 61.8),
            CdlCluster(6.3065, -16.0, 93.3, 5.6, 93.0, 51.9),
            CdlCluster(6.6374, -15.7, 106.6, 0.7, 92.9, 61.7),
            CdlCluster(7.0427, -21.6, 119.5, -21.9, 105.2, 58.0),
            CdlCluster(8.6523, -22.8, -123.8, 33.6, 107.8, 57.0),
        ),
        c_asd_deg=2.0,
        c_asa_deg=15.0,
        c_zsd_deg=3.0,
        c_zsa_deg=7.0,
        xpr_db=7.0,
    ),
    "CDL-D": CdlProfile(
        clusters=(
            CdlCluster(0.0000, -0.2, 0.0, -180.0, 98.5, 81.5, "LOS"),
            CdlCluster(0.0000, -13.5, 0.0, -180.0, 98.5, 81.5),
            CdlCluster(0.0350, -18.8, 89.2, 89.2, 85.5, 86.9),
            CdlCluster(0.6120, -21.0, 89.2, 89.2, 85.5, 86.9),
            CdlCluster(1.3630, -22.8, 89.2, 89.2, 85.5, 86.9),
            CdlCluster(1.4050, -17.9, 13.0, 163.0, 97.5, 79.4),
            CdlCluster(1.8040, -20.1, 13.0, 163.0, 97.5, 79.4),
            CdlCluster(2.5960, -21.9, 13.0, 163.0, 97.5, 79.4),
            CdlCluster(1.7750, -22.9, 34.6, -137.0, 98.5, 78.2),
            CdlCluster(4.0420, -27.8, -64.5, 74.5, 88.4, 73.6),
            CdlCluster(7.9370, -23.6, -32.9, 127.7, 91.3, 78.3),
            CdlCluster(9.4240, -24.8, 52.6, -119.6, 103.8, 87.0),
            CdlCluster(9.7080, -30.0, -132.1, -9.1, 80.3, 70.6),
            CdlCluster(12.5250, -27.7, 77.2, -83.8, 86.5, 72.9),
        ),
        c_asd_deg=5.0,
        c_asa_deg=8.0,
        c_zsd_deg=3.0,
        c_zsa_deg=3.0,
        xpr_db=11.0,
    ),
    "CDL-E": CdlProfile(
        clusters=(
            CdlCluster(0.0000, -0.03, 0.0, -180.0, 99.6, 80.4, "LOS"),
            CdlCluster(0.0000, -22.03, 0.0, -180.0, 99.6, 80.4),
            CdlCluster(0.5133, -15.8, 57.5, 18.2, 104.2, 80.4),
            CdlCluster(0.5440, -18.1, 57.5, 18.2, 104.2, 80.4),
            CdlCluster(0.5630, -19.8, 57.5, 18.2, 104.2, 80.4),
            CdlCluster(0.5440, -22.9, -20.1, 101.8, 99.4, 80.8),
            CdlCluster(0.7112, -22.4, 16.2, 112.9, 100.8, 86.3),
            CdlCluster(1.9092, -18.6, 9.3, -155.5, 98.8, 82.7),
            CdlCluster(1.9293, -20.8, 9.3, -155.5, 98.8, 82.7),
            CdlCluster(1.9589, -22.6, 9.3, -155.5, 98.8, 82.7),
            CdlCluster(2.6426, -22.3, 19.0, -143.3, 100.8, 82.9),
            CdlCluster(3.7136, -25.6, 32.7, -94.7, 96.4, 88.0),
            CdlCluster(5.4524, -20.2, 0.5, 147.0, 98.9, 81.0),
            CdlCluster(12.0034, -29.8, 55.9, -36.2, 95.6, 88.6),
            CdlCluster(20.6519, -29.2, 57.6, -26.0, 104.6, 78.3),
        ),
        c_asd_deg=5.0,
        c_asa_deg=11.0,
        c_zsd_deg=3.0,
        c_zsa_deg=7.0,
        xpr_db=8.0,
    ),
}


CDL_LOS_K_DB = {"CDL-D": 13.3, "CDL-E": 22.0}
