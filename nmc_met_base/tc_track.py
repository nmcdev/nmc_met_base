"""
===========================================
    台风路径追踪算法实现说明文档 （Created by DaiKan）
===========================================

本程序实现了基于数值模式输出的台风路径追踪算法。该算法用于识别并跟踪热带气旋（TC）的轨迹，
通过使用海平面气压、涡度、温度等物理量场数据，根据一定的准则识别和追踪台风位置。

### 主要功能：
1. **TC识别**：
    - 通过分析数值模式输出的海平面气压场（MSLP）、850hPa涡度场、850-200hPa厚度场等特征，自动识别热带气旋的位置。
    - 通过搜索低压中心、气旋环流和温暖核心特征来确认是否存在热带气旋。

2. **TC路径追踪**：
    - 基于初始台风位置，使用外推法和引导气流法计算台风未来位置，并在每个6小时时次更新台风位置。
    - 外推法：通过台风的过去位置进行线性外推。
    - 引导气流法：根据850、700、500和200hPa的加权平均风速估算未来位置。

3. **终止条件**：
    - 当台风的海平面气压超过1015 hPa、850hPa涡度过低、风速过低或台风进入高地形区域时，程序会停止追踪。

### 输入：
程序使用的输入数据是数值模式预测的多维数组，具体格式如下：
- `mslp_data`：海平面气压场，形状为 \[Nx, Ny, Nt\]。
- `vorticity_data`：850hPa 涡度场，形状为 \[Nx, Ny, Nt\]。
- `thickness_data`：850-200hPa 厚度场，形状为 \[Nx, Ny, Nt\]。
- `u850_data, v850_data`：850hPa 风场，形状为 \[Nx, Ny, Nt\]，包括 u 和 v 分量。
- `u700_data, v700_data`：700hPa 风场，形状为 \[Nx, Ny, Nt\]，包括 u 和 v 分量。
- `u500_data, v500_data`：500hPa 风场，形状为 \[Nx, Ny, Nt\]，包括 u 和 v 分量。
- `u200_data, v200_data`：200hPa 风场，形状为 \[Nx, Ny, Nt\]，包括 u 和 v 分量。
- `lat_grid, lon_grid`：网格的纬度和经度，形状为 \[Nx, Ny\]。

### 输出：
程序将返回一个包含每个时次台风位置信息的列表，每个元素是一个字典，包含以下字段：
- `time`：当前时刻，来自 `times` 列表。
- `lat`：台风当前的纬度。
- `lon`：台风当前的经度。
- `mslp`：海平面气压。
- `vorticity_850`：850hPa涡度。
- `remark`：备注信息，指示台风状态（如“成功跟踪”或“终止”）。

### 使用方法：
1. 设置起始台风位置 `initial_lat` 和 `initial_lon`。
2. 准备数值模式的预测数据 `mslp_data`、`vorticity_data`、`thickness_data`、风场数据等。
3. 定义时间步长列表 `times`，每个时刻的数据将被用于台风位置预测。
4. 调用 `track_tc` 函数开始追踪。

### 示例：
```python
track_results = track_tc(
    initial_lat=5.0, initial_lon=105.0,
    lat_grid=lat_grid_2d, lon_grid=lon_grid_2d,
    mslp_data=mslp_data, vorticity_data=vorticity_data,
    thickness_data=thickness_data,
    u850_data=u850_data, v850_data=v850_data,
    u700_data=u700_data, v700_data=v700_data,
    u500_data=u500_data, v500_data=v500_data,
    u200_data=u200_data, v200_data=v200_data,
    times=times
)

### 说明文档更新：
1. **版本日志**：
   - **v1.0.0**：初始版本的发布，功能包括台风识别、路径追踪、引导气流法和终止条件。
   - **v1.1.0**：增强了容错机制和数据一致性检查，并添加了日志与警告信息，确保程序在业务系统中运行时更加稳定和可靠。
"""

import numpy as np
import warnings
from math import radians, sin, cos, asin, sqrt

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    计算球面两点 (lat1, lon1) 和 (lat2, lon2) 的 haversine 大圆距离，单位 km。
    
    参数：
    ----------
    lat1, lon1 : float
        第一个点的纬度、经度 (单位: 度)
    lat2, lon2 : float
        第二个点的纬度、经度 (单位: 度)

    返回：
    ----------
    distance_km : float
        两点之间的球面距离 (单位: km)
    """
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = (sin(dlat/2))**2 + cos(radians(lat1))*cos(radians(lat2))*(sin(dlon/2))**2
    c = 2 * asin(sqrt(a))
    distance_km = R * c
    return distance_km

def _get_local_region_mask(lat_grid, lon_grid, lat_c, lon_c, radius):
    """
    在经纬网格 (lat_grid, lon_grid) 上，以 (lat_c, lon_c) 为中心，
    先用 bounding box 粗筛，再用 haversine 距离做精确筛选，得到在 radius 范围内的mask。

    参数:
    ----------
    lat_grid : 2D float array, shape (Nx, Ny)
    lon_grid : 2D float array, shape (Nx, Ny)
    lat_c, lon_c : float
    radius : float (km)

    返回:
    ----------
    mask : 2D bool array, shape (Nx, Ny)
    """
    if lat_grid.shape != lon_grid.shape:
        warnings.warn("lat_grid 与 lon_grid 维度不匹配，返回空 mask")
        return np.zeros_like(lat_grid, dtype=bool)

    Nx, Ny = lat_grid.shape
    if Nx == 0 or Ny == 0:
        warnings.warn("空网格，返回空 mask")
        return np.zeros_like(lat_grid, dtype=bool)

    # 粗筛: 1 度纬度约 111 km, 经度约 111*cos(lat_c)
    lat_deg_threshold = radius / 111.0
    cos_lat = np.cos(np.radians(lat_c))
    if abs(cos_lat) < 1e-6:
        # 极地附近特殊处理
        lon_deg_threshold = 180.0
    else:
        lon_deg_threshold = radius / (111.0 * abs(cos_lat))

    lat_min = lat_c - lat_deg_threshold
    lat_max = lat_c + lat_deg_threshold
    lon_min = lon_c - lon_deg_threshold
    lon_max = lon_c + lon_deg_threshold

    rough_mask = (
        (lat_grid >= lat_min) & (lat_grid <= lat_max) &
        (lon_grid >= lon_min) & (lon_grid <= lon_max)
    )
    if not np.any(rough_mask):
        return rough_mask

    # 对粗筛区域内再做 haversine 距离精确判断
    lat_candidates = lat_grid[rough_mask]
    lon_candidates = lon_grid[rough_mask]
    R = 6371.0
    dlat = np.radians(lat_candidates - lat_c)
    dlon = np.radians(lon_candidates - lon_c)
    a = np.sin(dlat/2.0)**2 + cos(radians(lat_c))*cos(np.radians(lat_candidates))*np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    distances_km = R * c

    fine_mask_local = distances_km <= radius

    mask = np.zeros_like(rough_mask, dtype=bool)
    local_indices = np.where(rough_mask)
    mask[local_indices] = fine_mask_local

    return mask

def find_mslp_min_center(lat_grid, lon_grid, mslp_field_2d, lat_c, lon_c, radius=445.0):
    """
    在 (lat_c, lon_c) 附近, 在 2D 海平面气压场 mslp_field_2d (shape Nx, Ny) 中搜索最低中心。

    返回:
    ----------
    center_found : bool
    (min_lat, min_lon) : (float, float) or (None, None)
    """
    if mslp_field_2d.shape != lat_grid.shape:
        warnings.warn("mslp_field_2d 与网格维度不匹配")
        return False, (None, None)

    mask = _get_local_region_mask(lat_grid, lon_grid, lat_c, lon_c, radius)
    if not np.any(mask):
        return False, (None, None)

    local_mslp = mslp_field_2d[mask]
    idx_min = np.argmin(local_mslp)
    full_indices = np.where(mask)
    min_lat = lat_grid[full_indices][idx_min]
    min_lon = lon_grid[full_indices][idx_min]
    return True, (min_lat, min_lon)

def find_local_vorticity_max(lat_grid, lon_grid, vort_field_2d, lat_c, lon_c, radius=278.0):
    """
    在 2D 涡度场 vort_field_2d (shape Nx, Ny) 中，搜索半径 278km 内的最大正涡度点(北半球示例)。
    """
    if vort_field_2d.shape != lat_grid.shape:
        warnings.warn("vort_field_2d 与网格维度不匹配")
        return False, (None, None)

    mask = _get_local_region_mask(lat_grid, lon_grid, lat_c, lon_c, radius)
    if not np.any(mask):
        return False, (None, None)

    local_vort = vort_field_2d[mask]
    idx_max = np.argmax(local_vort)
    max_vort = local_vort[idx_max]

    # 简单阈值
    if max_vort < 5e-5:
        return False, (None, None)

    full_indices = np.where(mask)
    vort_lat = lat_grid[full_indices][idx_max]
    vort_lon = lon_grid[full_indices][idx_max]
    return True, (vort_lat, vort_lon)

def find_local_thickness_max(lat_grid, lon_grid, thick_field_2d, lat_c, lon_c, radius=278.0):
    """
    在 2D 厚度场 thick_field_2d (shape Nx, Ny) 中，搜索半径 278km 内的最大值。
    """
    if thick_field_2d.shape != lat_grid.shape:
        warnings.warn("thick_field_2d 与网格维度不匹配")
        return False, (None, None)

    mask = _get_local_region_mask(lat_grid, lon_grid, lat_c, lon_c, radius)
    if not np.any(mask):
        return False, (None, None)

    local_thick = thick_field_2d[mask]
    idx_max = np.argmax(local_thick)
    full_indices = np.where(mask)
    thick_lat = lat_grid[full_indices][idx_max]
    thick_lon = lon_grid[full_indices][idx_max]
    return True, (thick_lat, thick_lon)

def check_tc_criteria(lat_grid, lon_grid,
                      mslp_field_2d, vort_field_2d, thick_field_2d,
                      lat_c, lon_c,
                      radius_mslp=445.0, radius_vort=278.0, radius_thick=278.0):
    """
    在给定位置 (lat_c, lon_c) 附近, 分别搜索:
      1) MSLP最低中心 (半径445km)
      2) 850hPa 涡度最大值 (正涡度, 半径278km)
      3) 850-200hPa 厚度最大值 (半径278km)

    参数:
    ----------
    lat_grid, lon_grid : shape (Nx, Ny)
    mslp_field_2d : shape (Nx, Ny)
    vort_field_2d : shape (Nx, Ny)
    thick_field_2d : shape (Nx, Ny)
    ...
    """
    # 1) 找最低气压
    found_mslp, (min_lat, min_lon) = find_mslp_min_center(
        lat_grid, lon_grid, mslp_field_2d, lat_c, lon_c, radius_mslp
    )
    if not found_mslp:
        return False, None

    # 2) 找涡度极大值
    found_vort, _ = find_local_vorticity_max(
        lat_grid, lon_grid, vort_field_2d, min_lat, min_lon, radius_vort
    )
    if not found_vort:
        return False, None

    # 3) 找厚度最大值
    found_thick, _ = find_local_thickness_max(
        lat_grid, lon_grid, thick_field_2d, min_lat, min_lon, radius_thick
    )
    if not found_thick:
        return False, None

    return True, (min_lat, min_lon)

def compute_steering_flow(lat_grid, lon_grid,
                          u850_2d, v850_2d,
                          u700_2d, v700_2d,
                          u500_2d, v500_2d,
                          u200_2d, v200_2d,
                          lat_c, lon_c,
                          radius=278.0,
                          weights=None):
    """
    计算引导气流，输入各层 2D 风场 (Nx, Ny), 在 (lat_c, lon_c) 附近做加权平均。
    """
    if weights is None:
        weights = [0.4, 0.3, 0.2, 0.1]

    # 基础维度检查
    for arr in [u850_2d, v850_2d, u700_2d, v700_2d, u500_2d, v500_2d, u200_2d, v200_2d]:
        if arr.shape != lat_grid.shape:
            warnings.warn("风场与网格维度不匹配，返回 (0,0)")
            return 0.0, 0.0

    mask = _get_local_region_mask(lat_grid, lon_grid, lat_c, lon_c, radius)
    if not np.any(mask):
        return 0.0, 0.0

    w_850, w_700, w_500, w_200 = weights
    local_u850 = u850_2d[mask]
    local_v850 = v850_2d[mask]
    local_u700 = u700_2d[mask]
    local_v700 = v700_2d[mask]
    local_u500 = u500_2d[mask]
    local_v500 = v500_2d[mask]
    local_u200 = u200_2d[mask]
    local_v200 = v200_2d[mask]

    mean_u850 = np.mean(local_u850)
    mean_v850 = np.mean(local_v850)
    mean_u700 = np.mean(local_u700)
    mean_v700 = np.mean(local_v700)
    mean_u500 = np.mean(local_u500)
    mean_v500 = np.mean(local_v500)
    mean_u200 = np.mean(local_u200)
    mean_v200 = np.mean(local_v200)

    u_steer = (mean_u850 * w_850 + mean_u700 * w_700 +
               mean_u500 * w_500 + mean_u200 * w_200)
    v_steer = (mean_v850 * w_850 + mean_v700 * w_700 +
               mean_v500 * w_500 + mean_v200 * w_200)

    return u_steer, v_steer

def check_termination_conditions(mslp, vorticity_850, lat, lon,
                                terrain_height=None, wind10m=None,
                                lat0=None, lon0=None,
                                max_pressure=1015.0,
                                min_vort=5e-5,
                                max_land_wind=8.0,
                                max_terrain_height=1000.0,
                                max_distance=278.0):
    """
    检查是否满足停止追踪的条件 (任意一个满足则停止追踪)：
      - 海平面气压 > max_pressure
      - 850hPa 涡度 < min_vort
      - 10m 风速 < max_land_wind (若在陆地)
      - 地形高 > max_terrain_height 且偏移距离 > max_distance
    """
    if mslp is None or np.isnan(mslp):
        warnings.warn("MSLP 数据异常")
        return True

    # 1) 氣压
    if mslp > max_pressure:
        return True

    # 2) 涡度
    if vorticity_850 is None or np.isnan(vorticity_850) or vorticity_850 < min_vort:
        return True

    # 3) 10m风速 (简化)
    if wind10m is not None and wind10m < max_land_wind:
        return True

    # 4) 高地形 & 偏移
    if terrain_height is not None and terrain_height > max_terrain_height:
        if lat0 is not None and lon0 is not None:
            dist = haversine_distance(lat0, lon0, lat, lon)
            if dist > max_distance:
                return True

    return False

def track_tc(initial_lat, initial_lon,
             lat_grid, lon_grid,
             mslp_data, vorticity_data, thickness_data,
             u850_data, v850_data,
             u700_data, v700_data,
             u500_data, v500_data,
             u200_data, v200_data,
             times,
             max_steps=50,
             consecutive_miss_tolerance=2):
    """
    台风追踪主函数：输入 3D 物理量数据 mslp_data, vorticity_data, thickness_data 等，形状均为 (Nx, Ny, Nt)。
    lat_grid, lon_grid 则为 (Nx, Ny)。
    times: 长度为 Nt 的时间序列。

    参数:
    ----------
    initial_lat, initial_lon : float
        起始时刻台风定位 (度)
    lat_grid, lon_grid : 2D float array, shape (Nx, Ny)
        网格坐标 (度)
    mslp_data, vorticity_data, thickness_data : 3D float array, shape (Nx, Ny, Nt)
        海平面气压、850hPa涡度、850-200hPa厚度
    u850_data, v850_data : 3D float array, shape (Nx, Ny, Nt)
        850hPa 风场
    u700_data, v700_data : 3D float array, shape (Nx, Ny, Nt)
        700hPa 风场
    u500_data, v500_data : 3D float array, shape (Nx, Ny, Nt)
        500hPa 风场
    u200_data, v200_data : 3D float array, shape (Nx, Ny, Nt)
        200hPa 风场
    times : 1D array or list, length Nt
        时间维度 (可为 [0, 6, 12, 18, ...] 小时 或 datetime 对象)
    max_steps : int
        最大追踪步数
    consecutive_miss_tolerance : int
        允许连续识别失败的最大次数

    返回:
    ----------
    track_results : list of dict
        每时次识别到的台风信息
    """
    # 1) 基础检查
    Nx, Ny, Nt = mslp_data.shape
    if vorticity_data.shape != (Nx, Ny, Nt) or thickness_data.shape != (Nx, Ny, Nt):
        raise ValueError("vorticity_data 或 thickness_data 与 mslp_data 的形状不匹配")

    if (u850_data.shape != (Nx, Ny, Nt) or v850_data.shape != (Nx, Ny, Nt) or
        u700_data.shape != (Nx, Ny, Nt) or v700_data.shape != (Nx, Ny, Nt) or
        u500_data.shape != (Nx, Ny, Nt) or v500_data.shape != (Nx, Ny, Nt) or
        u200_data.shape != (Nx, Ny, Nt) or v200_data.shape != (Nx, Ny, Nt)):
        raise ValueError("风场数据形状不匹配 (Nx, Ny, Nt)")

    if lat_grid.shape != (Nx, Ny) or lon_grid.shape != (Nx, Ny):
        raise ValueError("lat_grid, lon_grid 形状应为 (Nx, Ny)")

    if len(times) != Nt:
        raise ValueError("times 长度与数据第三维度 Nt 不符")

    # 2) 开始追踪
    track_results = []
    current_lat, current_lon = initial_lat, initial_lon
    prev_lat, prev_lon = None, None
    missed_count = 0

    for i in range(Nt):
        if i >= max_steps:
            warnings.warn("已达到 max_steps，终止追踪")
            break

        # 取当前时次的 2D 切片
        mslp_2d = mslp_data[:, :, i]
        vort_2d = vorticity_data[:, :, i]
        thick_2d = thickness_data[:, :, i]

        u850_2d = u850_data[:, :, i]
        v850_2d = v850_data[:, :, i]
        u700_2d = u700_data[:, :, i]
        v700_2d = v700_data[:, :, i]
        u500_2d = u500_data[:, :, i]
        v500_2d = v500_data[:, :, i]
        u200_2d = u200_data[:, :, i]
        v200_2d = v200_data[:, :, i]

        # 时间信息
        t = times[i]

        if i == 0:
            # 第 0 个时次：用 (initial_lat, initial_lon) 附近搜索
            found, center_pos = check_tc_criteria(
                lat_grid, lon_grid,
                mslp_2d, vort_2d, thick_2d,
                current_lat, current_lon
            )
            if found:
                current_lat, current_lon = center_pos
                # 估算网格位置
                center_idx_lat = np.argmin(np.abs(lat_grid[:, 0] - current_lat))
                center_idx_lon = np.argmin(np.abs(lon_grid[0, :] - current_lon))
                current_mslp = mslp_2d[center_idx_lat, center_idx_lon]
                current_vort = vort_2d[center_idx_lat, center_idx_lon]

                # 终止条件检查
                terminate = check_termination_conditions(
                    current_mslp, current_vort, current_lat, current_lon
                )
                if terminate:
                    track_results.append({
                        "time": t,
                        "lat": current_lat,
                        "lon": current_lon,
                        "mslp": current_mslp,
                        "vorticity_850": current_vort,
                        "remark": "Terminated at initial step"
                    })
                    break

                track_results.append({
                    "time": t,
                    "lat": current_lat,
                    "lon": current_lon,
                    "mslp": current_mslp,
                    "vorticity_850": current_vort,
                    "remark": "Initial detection"
                })
            else:
                missed_count += 1
                track_results.append({
                    "time": t,
                    "lat": current_lat,
                    "lon": current_lon,
                    "mslp": np.nan,
                    "vorticity_850": np.nan,
                    "remark": "No TC found at initial time"
                })
                if missed_count > consecutive_miss_tolerance:
                    warnings.warn(f"连续 {missed_count} 次未能识别到TC, 终止追踪")
                    break
        else:
            # 后续时次
            if prev_lat is not None and prev_lon is not None:
                dlat = current_lat - prev_lat
                dlon = current_lon - prev_lon
                lat_extrap = current_lat + dlat
                lon_extrap = current_lon + dlon
            else:
                lat_extrap = current_lat
                lon_extrap = current_lon

            # 引导气流
            u_steer, v_steer = compute_steering_flow(
                lat_grid, lon_grid,
                u850_2d, v850_2d,
                u700_2d, v700_2d,
                u500_2d, v500_2d,
                u200_2d, v200_2d,
                current_lat, current_lon
            )

            hours = 6  # 每个时次间隔6小时
            dist_km_u = u_steer * hours * 3600 / 1000.0
            dist_km_v = v_steer * hours * 3600 / 1000.0
            cos_lat = np.cos(np.radians(current_lat)) or 1e-6
            dlat_steer = dist_km_v / 111.0
            dlon_steer = dist_km_u / (111.0 * cos_lat)

            lat_steer = current_lat + dlat_steer
            lon_steer = current_lon + dlon_steer

            # 两种方法平均
            lat_guess = 0.5 * (lat_extrap + lat_steer)
            lon_guess = 0.5 * (lon_extrap + lon_steer)

            found, center_pos = check_tc_criteria(
                lat_grid, lon_grid,
                mslp_2d, vort_2d, thick_2d,
                lat_guess, lon_guess
            )
            if found:
                missed_count = 0
                new_lat, new_lon = center_pos
                center_idx_lat = np.argmin(np.abs(lat_grid[:, 0] - new_lat))
                center_idx_lon = np.argmin(np.abs(lon_grid[0, :] - new_lon))
                current_mslp = mslp_2d[center_idx_lat, center_idx_lon]
                current_vort = vort_2d[center_idx_lat, center_idx_lon]

                # 终止条件
                terminate = check_termination_conditions(
                    current_mslp, current_vort, new_lat, new_lon,
                    lat0=current_lat, lon0=current_lon
                )
                if terminate:
                    track_results.append({
                        "time": t,
                        "lat": new_lat,
                        "lon": new_lon,
                        "mslp": current_mslp,
                        "vorticity_850": current_vort,
                        "remark": "TC terminated"
                    })
                    break

                prev_lat, prev_lon = current_lat, current_lon
                current_lat, current_lon = new_lat, new_lon

                track_results.append({
                    "time": t,
                    "lat": current_lat,
                    "lon": current_lon,
                    "mslp": current_mslp,
                    "vorticity_850": current_vort,
                    "remark": "TC tracked"
                })
            else:
                missed_count += 1
                track_results.append({
                    "time": t,
                    "lat": lat_guess,
                    "lon": lon_guess,
                    "mslp": np.nan,
                    "vorticity_850": np.nan,
                    "remark": "TC not found"
                })
                if missed_count > consecutive_miss_tolerance:
                    warnings.warn(f"连续 {missed_count} 次未能识别到TC, 终止追踪")
                    break

    return track_results

# # ------------------- 示例主程序 ------------------- #
# if __name__ == "__main__":
#     """
#     实际业务中，需要从 NetCDF/GRIB 文件读取并插值后，
#     生成下列 lat_grid, lon_grid, mslp_data, vorticity_data, thickness_data, wind... 变量。
#     这里仅用随机数做一个示例，形状为 (Nx, Ny, Nt)。
#     """

#     # 假设 Nx=50, Ny=60, Nt=5 (5个时间步)
#     Nx, Ny, Nt = 50, 60, 5

#     # 构造网格坐标: shape (Nx, Ny)
#     # 注意此处 Nx 对应 x方向(如经度), Ny 对应 y方向(如纬度)，仅做演示
#     x_vals = np.linspace(100, 110, Nx)   # 经度方向
#     y_vals = np.linspace(0, 10, Ny)      # 纬度方向
#     # 让 lat_grid[i, j], lon_grid[i, j]
#     # 其中 i in [0..Nx-1], j in [0..Ny-1]
#     # 这与多数 "lat, lon" 定义反过来了(通常 lat 是第0维, lon 是第1维)，
#     # 仅为演示 shape (Nx, Ny) 的一致性。
#     lon_grid_2d, lat_grid_2d = np.meshgrid(x_vals, y_vals, indexing='ij')  

#     # 生成 3D 数据 (Nx, Ny, Nt)
#     mslp_data = 1000 + 5*np.random.rand(Nx, Ny, Nt)
#     vorticity_data = 5e-5 + 1e-5*np.random.rand(Nx, Ny, Nt)
#     thickness_data = 5800 + 10*np.random.rand(Nx, Ny, Nt)

#     u850_data = np.random.rand(Nx, Ny, Nt)
#     v850_data = np.random.rand(Nx, Ny, Nt)
#     u700_data = np.random.rand(Nx, Ny, Nt)
#     v700_data = np.random.rand(Nx, Ny, Nt)
#     u500_data = np.random.rand(Nx, Ny, Nt)
#     v500_data = np.random.rand(Nx, Ny, Nt)
#     u200_data = np.random.rand(Nx, Ny, Nt)
#     v200_data = np.random.rand(Nx, Ny, Nt)

#     # 时间序列 (简单用 0,6,12,18,... 小时表示)
#     times = np.array([0, 6, 12, 18, 24])

#     # 假设初始台风位置(在网格中央附近)
#     initial_lat = 5.0
#     initial_lon = 105.0

#     track = track_tc(
#         initial_lat, initial_lon,
#         lat_grid_2d, lon_grid_2d,
#         mslp_data, vorticity_data, thickness_data,
#         u850_data, v850_data,
#         u700_data, v700_data,
#         u500_data, v500_data,
#         u200_data, v200_data,
#         times
#     )

#     print("=== TC Tracking Results ===")
#     for rec in track:
#         print(rec)
