# _*_ coding: utf-8 _*_

# Copyright (c) 2019 NMC Developers.
# Distributed under the terms of the GPL V3 License.

"""
Polygon, line and point algorithm.

refer to:
https://github.com/inasafe/python-safe/blob/master/safe/engine/polygon.py
"""

import numpy as np
from random import uniform, seed as seed_function
from nmc_met_base.numeric import ensure_numeric


def point_inside_polygon(x, y, points):
    """
    Return True if a coordinate (x, y) is inside a polygon defined by
    a list of verticies [(x1, y1), (x2, x2), ... , (xN, yN)].

    Reference: http://www.ariel.com.au/a/python-point-int-poly.html
    """
    n = len(points)
    inside = False
    p1x, p1y = points[0]
    for i in range(1, n + 1):
        p2x, p2y = points[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def separate_points_by_polygon(points, polygon, closed=True):
    """
    "Determine whether points are inside or outside a polygon.
    The vertices may be listed clockwise or counterclockwise and
       the first point may optionally be repeated.
       Polygons do not need to be convex.
       Polygons can have holes in them and points inside a hole is
       regarded as being outside the polygon.
    Algorithm is based on work by Darel Finley,
    http://www.alienryderflex.com/polygon/

    :param points: Tuple of (x, y) coordinates, or list of tuples.
    :param polygon: list of vertices of polygon
    :param closed: (optional) determine whether points on boundary should be
                   regarded as belonging to the polygon (closed = True)
    :return: indices: array of same length as points with indices of points
                      falling inside the polygon listed from the beginning
                      and indices of points falling outside listed from
                      the end.
             count: count of points falling inside the polygon
             The indices of points inside are obtained as indices[:count]
             The indices of points outside are obtained as indices[count:]
    :Examples:
        U = [[0,0], [1,0], [1,1], [0,1]]  # Unit square

        separate_points_by_polygon( [[0.5, 0.5], [1, -0.5], [0.3, 0.2]], U)
        will return the indices [0, 2, 1] and count == 2 as only the first
        and the last point are inside the unit square
    """

    # check inputs
    #
    if len(polygon.shape) != 2:
        raise Exception('Polygon array must be a 2d array of vertices.')
    if not 0 < len(points.shape) < 3:
        raise Exception('Points array must be 1 or 2 dimensional.')
    # Only one point was passed in. Convert to array of points.
    if len(points.shape) == 1:
        points = np.reshape(points, (1, 2))
    if points.shape[1] != 2:
        raise Exception('Point array must have two columns (x,y).')
    if len(points.shape) != 2:
        raise Exception('Points array must have two columns.')

    _ = polygon.shape[0]  # Number of vertices in polygon
    _ = points.shape[0]  # Number of points

    # indices = numpy.zeros(M, numpy.int)
    indices, count = _separate_points_by_polygon(
        points, polygon, closed=closed)

    return indices, count


def _separate_points_by_polygon(points, polygon, closed):
    """
    Underlying algorithm to partition point according to polygon.

    :param points:
    :param polygon:
    :param closed:
    :return:
    """

    # Suppress numpy warnings (as we'll be dividing by zero)
    original_numpy_settings = np.seterr(invalid='ignore', divide='ignore')

    rtol = 0.0
    atol = 0.0

    # Get polygon extents to quickly rule out points that
    # are outside its bounding box
    minpx = min(polygon[:, 0])
    maxpx = max(polygon[:, 0])
    minpy = min(polygon[:, 1])
    maxpy = max(polygon[:, 1])

    M = points.shape[0]
    N = polygon.shape[0]

    x = points[:, 0]
    y = points[:, 1]

    # Vector to return sorted indices (inside first, then outside)
    indices = np.zeros(M, np.int)

    # Vector keeping track of which points are inside
    inside = np.zeros(M, dtype=np.int)  # All assumed outside initially

    # Mask for points can be considered for inclusion
    candidates = np.ones(M, dtype=np.bool)  # All True initially

    # Only work on those that are inside polygon bounding box
    outside_box = (x > maxpx) + (x < minpx) + (y > maxpy) + (y < minpy)
    inside_box = -outside_box
    candidates *= inside_box

    # Don't continue if all points are outside bounding box
    if not np.sometrue(candidates):
        return np.arange(M)[::-1], 0

    # Find points on polygon boundary
    for i in range(N):
        # Loop through polygon edges
        j = (i + 1) % N
        edge = [polygon[i, :], polygon[j, :]]

        # Select those that are on the boundary
        boundary_points = point_on_line(points, edge, rtol, atol)

        if closed:
            inside[boundary_points] = 1
        else:
            inside[boundary_points] = 0

        # Remove boundary point from further analysis
        candidates[boundary_points] = False

    # Algorithm for finding points inside polygon
    for i in range(N):
        # Loop through polygon edges
        j = (i + 1) % N
        px_i, py_i = polygon[i, :]
        px_j, py_j = polygon[j, :]

        # Intersection formula
        sigma = (y - py_i) / (py_j - py_i) * (px_j - px_i)
        seg_i = (py_i < y) * (py_j >= y)
        seg_j = (py_j < y) * (py_i >= y)
        mask = (px_i + sigma < x) * (seg_i + seg_j) * candidates

        inside[mask] = 1 - inside[mask]

    # Restore numpy warnings
    np.seterr(**original_numpy_settings)

    # Record point as either inside or outside
    inside_index = np.sum(inside)  # How many points are inside
    if inside_index == 0:
        # Return all indices as points outside
        # and code that depends on this order.
        return np.arange(M)[::-1], 0

    # Indices of inside points
    indices[:inside_index] = np.where(inside)[0]
    # Indices of outside points (reversed...)
    indices[inside_index:] = np.where(1 - inside)[0][::-1]

    return indices, inside_index


def point_on_line(points, line, rtol=1.0e-5, atol=1.0e-8):
    """
    Determine if a point is on a line segment.
      Line can be degenerate and function still works to discern coinciding
      points from non-coinciding.
      Tolerances rtol and atol are used with numpy.allclose()

    :param points: Coordinates of either,
                   * one point given by sequence [x, y]
                   * multiple points given by list of points or Nx2 array
    :param line: Endpoint coordinates [[x0, y0], [x1, y1]] or
                 the equivalent 2x2 numeric array with each row corresponding
                 to a point.
    :param rtol: Relative error for how close a point must be to be accepted
    :param atol: Absolute error for how close a point must be to be accepted
    :return: True or False
    """

    # Prepare input data
    points = ensure_numeric(points)
    line = ensure_numeric(line)

    if len(points.shape) == 1:
        # One point only - make into 1 x 2 array
        points = points[np.newaxis, :]
        one_point = True
    else:
        one_point = False

    msg = 'Argument points must be either [x, y] or an Nx2 array of points'
    if len(points.shape) != 2:
        raise Exception(msg)
    if not points.shape[0] > 0:
        raise Exception(msg)
    if points.shape[1] != 2:
        raise Exception(msg)

    N = points.shape[0]  # Number of points

    x = points[:, 0]
    y = points[:, 1]
    x0, y0 = line[0]
    x1, y1 = line[1]

    # Vector from beginning of line to point
    a0 = x - x0
    a1 = y - y0

    # It's normal vector
    a_normal0 = a1
    a_normal1 = -a0

    # Vector parallel to line
    b0 = x1 - x0
    b1 = y1 - y0

    # Dot product
    nominator = abs(a_normal0 * b0 + a_normal1 * b1)
    denominator = b0 * b0 + b1 * b1

    # Determine if point vector is parallel to line up to a tolerance
    is_parallel = np.zeros(N, dtype=np.bool)  # All False
    is_parallel[nominator <= atol + rtol * denominator] = True

    # Determine for points parallel to line if they are within end points
    a0p = a0[is_parallel]
    a1p = a1[is_parallel]

    len_a = np.sqrt(a0p * a0p + a1p * a1p)
    len_b = np.sqrt(b0 * b0 + b1 * b1)
    cross = a0p * b0 + a1p * b1

    # Initialise result to all False
    result = np.zeros(N, dtype=np.bool)

    # Result is True only if a0 * b0 + a1 * b1 >= 0 and len_a <= len_b
    result[is_parallel] = (cross >= 0) * (len_a <= len_b)

    # Return either boolean scalar or boolean vector
    if one_point:
        return result[0]
    else:
        return result


def inside_polygon(points, polygon, closed=True):
    """Determine points inside a polygon
       Functions inside_polygon and outside_polygon have been defined in
       terms of separate_by_polygon which will put all inside indices in
       the first part of the indices array and outside indices in the last
       See separate_points_by_polygon for documentation
       points and polygon can be a geospatial instance,
       a list or a numeric array
    """

    indices, count = separate_points_by_polygon(points, polygon, closed=closed)

    # Return indices of points inside polygon
    return indices[:count]


def is_inside_polygon(point, polygon, closed=True):
    """Determine if one point is inside a polygon
    See inside_polygon for more details
    """

    indices = inside_polygon(point, polygon, closed)

    if indices.shape[0] == 1:
        return True
    else:
        return False


def is_outside_polygon(point, polygon, closed=True):
    """Determine if one point is outside a polygon
    See outside_polygon for more details
    """

    indices = outside_polygon(point, polygon, closed=closed)

    if indices.shape[0] == 1:
        return True
    else:
        return False


def outside_polygon(points, polygon, closed=True):
    """Determine points outside a polygon
       Functions inside_polygon and outside_polygon have been defined in
       terms of separate_by_polygon which will put all inside indices in
       the first part of the indices array and outside indices in the last
       See separate_points_by_polygon for documentation
    """

    indices, count = separate_points_by_polygon(points, polygon, closed=closed)

    # Return indices of points outside polygon
    if count == len(indices):
        # No points are outside
        return np.array([])
    else:
        # Return indices for points outside (reversed)
        return indices[count:][::-1]


def in_and_outside_polygon(points, polygon, closed=True):
    """Separate a list of points into two sets inside and outside a polygon
    Input
        points: (tuple, list or array) of coordinates
        polygon: Set of points defining the polygon
        closed: Set to True if points on boundary are considered
                to be 'inside' polygon
    Output
        inside: Array of points inside the polygon
        outside: Array of points outside the polygon
    See separate_points_by_polygon for more documentation
    """

    indices, count = separate_points_by_polygon(points, polygon, closed=closed)

    if count == len(indices):
        # No points are outside
        return indices[:count], []
    else:
        # Return indices for points inside and outside (reversed)
        return indices[:count], indices[count:][::-1]


def clip_lines_by_polygon(lines, polygon, closed=True):
    """
    Clip multiple lines by polygon.
    This is a wrapper around clip_line_by_polygon

    :param lines: Sequence of polylines: [[p0, p1, ...], [q0, q1, ...], ...]
                  where pi, qi, ... are point coordinates (x, y).
    :param polygon: list of vertices of polygon or the corresponding
                    numpy array
    :param closed: (optional) determine whether points on boundary should be
                   regarded as belonging to the polygon (closed = True)
                   or not (closed = False) - False is not recommended here
    :return: inside_line_segments: Clipped line segments that are
                                   inside polygon
             outside_line_segments: Clipped line segments that are
                                    outside polygon
    """

    _ = polygon.shape[0]  # Number of vertices in polygon
    _ = len(lines)  # Number of lines

    inside_line_segments = []
    outside_line_segments = []

    # Loop through lines
    for k in range(M):
        inside, outside = clip_line_by_polygon(
            lines[k], polygon, closed=closed)
        inside_line_segments += inside
        outside_line_segments += outside

    return inside_line_segments, outside_line_segments


def clip_line_by_polygon(line, polygon, closed=True):
    """
    Clip line segments by polygon.
    The assumptions listed in separate_points_by_polygon apply
    Output line segments are listed as separate lines i.e. not joined.

    :param line: Sequence of line nodes: [[x0, y0], [x1, y1], ...] or
                 the equivalent Nx2 numpy array
    :param polygon: list of vertices of polygon or the corresponding
                    numpy array
    :param closed: (optional) determine whether points on boundary should be
                   regarded as belonging to the polygon (closed = True)
                   or not (closed = False) - False is not recommended here
    :return: inside_lines: Clipped lines that are inside polygon
             outside_lines: Clipped lines that are outside polygon
             Both outputs take the form of lists of Nx2 line arrays

    :Example:
        U = [[0,0], [1,0], [1,1], [0,1]]  # Unit square
        # Simple horizontal fully intersecting line
        line = [[-1, 0.5], [2, 0.5]]
        inside_line_segments, outside_line_segments = \
            clip_line_by_polygon(line, polygon)
        print(np.allclose(inside_line_segments, [[[0, 0.5], [1, 0.5]]]))
        print(np.allclose(outside_line_segments,
                          [[[-1, 0.5], [0, 0.5]],
                           [[1, 0.5], [2, 0.5]]]))
    """

    # Get polygon extents to quickly rule out points that
    # are outside its bounding box
    minpx = min(polygon[:, 0])
    maxpx = max(polygon[:, 0])
    minpy = min(polygon[:, 1])
    maxpy = max(polygon[:, 1])

    N = polygon.shape[0]  # Number of vertices in polygon
    M = line.shape[0]  # Number of segments

    # Algorithm
    #
    # 1: Find all intersection points between line segments and polygon edges
    # 2: For each line segment
    #    * Calculate distance from first end point to each intersection point
    #    * Sort intersection points by distance
    #    * Cut segment into multiple segments
    # 3: For each new line segment
    #    * Calculate its midpoint
    #    * Determine if it is inside or outside clipping polygon

    # Loop through line segments
    inside_line_segments = []
    outside_line_segments = []

    for k in range(M - 1):
        p0 = line[k, :]
        p1 = line[k + 1, :]
        segment = [p0, p1]

        # -------------
        # Optimisation
        # -------------
        # Skip segments where both end points are outside polygon bounding box
        # and which don't intersect the bounding box

        # Multiple lines are clipped correctly by complex polygon ... ok
        # Ran 1 test in 187.759s
        # Ran 1 test in 12.517s
        segment_is_outside_bbox = True
        for p in [p0, p1]:
            x = p[0]
            y = p[1]
            if not (x > maxpx or x < minpx or y > maxpy or y < minpy):
                #  This end point is inside polygon bounding box
                segment_is_outside_bbox = False
                break

        # Does segment intersect polygon bounding box?
        corners = np.array([[minpx, minpy], [maxpx, minpy],
                            [maxpx, maxpy], [minpx, maxpy]])
        for i in range(3):
            edge = [corners[i, :], corners[i + 1, :]]
            status, value = intersection(segment, edge)
            if value is not None:
                # Segment intersects polygon bounding box
                segment_is_outside_bbox = False
                break
        # -----------------
        # End optimisation
        # -----------------

        # Separate segments that are inside from those outside
        if segment_is_outside_bbox:
            outside_line_segments.append(segment)
        else:
            # Intersect segment with all polygon edges
            # and decide for each sub-segment
            intersections = list(segment)  # Initialise with end points
            for i in range(N):
                # Loop through polygon edges
                j = (i + 1) % N
                edge = [polygon[i, :], polygon[j, :]]

                status, value = intersection(segment, edge)
                if status == 2:
                    # Collinear overlapping lines found
                    # Use midpoint of common segment
                    #              common segment directly
                    value = (value[0] + value[1]) / 2

                if value is not None:
                    # Record intersection point found
                    intersections.append(value)
                else:
                    pass

            # Loop through intersections for this line segment
            distances = {}
            P = len(intersections)
            for i in range(P):
                v = segment[0] - intersections[i]
                d = np.dot(v, v)
                distances[d] = intersections[i]  # Don't record duplicates

            # Sort by Schwarzian transform
            A = sorted(zip(distances.keys(), distances.values()))
            _, intersections = zip(*A)

            P = len(intersections)

            # Separate segments according to polygon
            for i in range(P - 1):
                segment = [intersections[i], intersections[i + 1]]
                midpoint = (segment[0] + segment[1]) / 2

                if is_inside_polygon(midpoint, polygon, closed=closed):
                    inside_line_segments.append(segment)
                else:
                    outside_line_segments.append(segment)

    # Rejoin adjacent segments and add to result lines
    inside_lines = join_line_segments(inside_line_segments)
    outside_lines = join_line_segments(outside_line_segments)

    return inside_lines, outside_lines


def join_line_segments(segments, rtol=1.0e-12, atol=1.0e-12):
    """
    Join adjacent line segments

    :param segments: List of distinct line segments [[p0, p1], [p2, p3], ...]
    :param rtol: Optional tolerances passed on to np.allclose
    :param atol: Optional tolerances passed on to np.allclose
    :return: list of Nx2 numpy arrays each corresponding to a continuous line
             formed from consecutive segments
    """

    lines = []

    if len(segments) == 0:
        return lines

    line = segments[0]
    for i in range(len(segments) - 1):
        if np.allclose(segments[i][1], segments[i + 1][0],
                       rtol=rtol, atol=atol):
            # Segments are adjacent
            line.append(segments[i + 1][1])
        else:
            # Segments are disjoint - current line finishes here
            lines.append(line)
            line = segments[i + 1]

    # Finish
    lines.append(line)

    # Return
    return lines


def populate_polygon(polygon, number_of_points, seed=None, exclude=None):
    """
    Populate given polygon with uniformly distributed points.

    :param polygon: list of vertices of polygon
    :param number_of_points: (optional) number of points
    :param seed: seed for random number generator (default=None)
    :param exclude: list of polygons (inside main polygon) from where points
                    should be excluded
    :return: points - list of points inside polygon

    :Examples:
       populate_polygon( [[0,0], [1,0], [1,1], [0,1]], 5 )
       will return five randomly selected points inside the unit square
    """

    # Find outer extent of polygon
    max_x = min_x = polygon[0][0]
    max_y = min_y = polygon[0][1]
    for point in polygon[1:]:
        x = point[0]
        if x > max_x:
            max_x = x
        if x < min_x:
            min_x = x

        y = point[1]
        if y > max_y:
            max_y = y
        if y < min_y:
            min_y = y

    # Generate random points until enough are in polygon
    seed_function(seed)
    points = []
    while len(points) < number_of_points:
        x = uniform(min_x, max_x)
        y = uniform(min_y, max_y)

        append = False
        if is_inside_polygon([x, y], polygon):
            append = True

            # Check exclusions
            if exclude is not None:
                for ex_poly in exclude:
                    if is_inside_polygon([x, y], ex_poly):
                        append = False

        if append is True:
            points.append([x, y])

    return points


def intersection(line0, line1, rtol=1.0e-12, atol=1.0e-12):
    """
    Returns intersecting point between two line segments.
    However, if parallel lines coincide partly (i.e. share a common segment),
    the line segment where lines coincide is returned.

    :param line0, line1: Each defined by two end points as in:
                         [[x0, y0], [x1, y1]]
                         A line can also be a 2x2 numpy array with each row
                         corresponding to a point.
    :param rtol, atol: Tolerances passed onto np.allclose
    :return: status, value - where status and value is interpreted as follows:
             status == 0: no intersection, value set to None.
             status == 1: intersection point found and returned in
                          value as [x, y].
             status == 2: Collinear overlapping lines found.
                          Value takes the form [[x0,y0], [x1,y1]] which is the
                          segment common to both lines.
             status == 3: Collinear non-overlapping lines. Value set to None.
             status == 4: Lines are parallel. Value set to None.
    """

    # Result functions used in intersection() below for possible states
    # of collinear lines. (p0,p1) defines line 0, (p2,p3) defines line 1.
    def lines_dont_coincide(p0, p1, p2, p3):
        return 3, None

    def lines_0_fully_included_in_1(p0, p1, p2, p3):
        return 2, np.array([p0, p1])

    def lines_1_fully_included_in_0(p0, p1, p2, p3):
        return 2, np.array([p2, p3])

    def lines_overlap_same_direction(p0, p1, p2, p3):
        return 2, np.array([p0, p3])

    def lines_overlap_same_direction2(p0, p1, p2, p3):
        return 2, np.array([p2, p1])

    def lines_overlap_opposite_direction(p0, p1, p2, p3):
        return 2, np.array([p0, p2])

    def lines_overlap_opposite_direction2(p0, p1, p2, p3):
        return 2, np.array([p3, p1])

    # This function called when an impossible state is found
    def lines_error(p1, p2, p3, p4):
        msg = ('Impossible state: p1=%s, p2=%s, p3=%s, p4=%s'
               % (str(p1), str(p2), str(p3), str(p4)))
        raise RuntimeError(msg)

    # Mapping to possible states for line intersection
    #
    #   0s1    0e1    1s0    1e0   # line 0 starts on 1, 0 ends 1,
    #                                1 starts 0, 1 ends 0
    collinearmap = {
        (False, False, False, False): lines_dont_coincide,
        (False, False, False, True): lines_error,
        (False, False, True, False): lines_error,
        (False, False, True, True): lines_1_fully_included_in_0,
        (False, True, False, False): lines_error,
        (False, True, False, True): lines_overlap_opposite_direction2,
        (False, True, True, False): lines_overlap_same_direction2,
        (False, True, True, True): lines_1_fully_included_in_0,
        (True, False, False, False): lines_error,
        (True, False, False, True): lines_overlap_same_direction,
        (True, False, True, False): lines_overlap_opposite_direction,
        (True, False, True, True): lines_1_fully_included_in_0,
        (True, True, False, False): lines_0_fully_included_in_1,
        (True, True, False, True): lines_0_fully_included_in_1,
        (True, True, True, False): lines_0_fully_included_in_1,
        (True, True, True, True): lines_0_fully_included_in_1}

    line0 = ensure_numeric(line0, np.float)
    line1 = ensure_numeric(line1, np.float)

    x0, y0 = line0[0, :]
    x1, y1 = line0[1, :]

    x2, y2 = line1[0, :]
    x3, y3 = line1[1, :]

    denom = (y3 - y2) * (x1 - x0) - (x3 - x2) * (y1 - y0)
    u0 = (x3 - x2) * (y0 - y2) - (y3 - y2) * (x0 - x2)
    u1 = (x2 - x0) * (y1 - y0) - (y2 - y0) * (x1 - x0)

    if np.allclose(denom, 0.0, rtol=rtol, atol=atol):
        # Lines are parallel - check if they are collinear
        if np.allclose([u0, u1], 0.0, rtol=rtol, atol=atol):
            # We now know that the lines are collinear
            state = (point_on_line([x0, y0], line1, rtol=rtol, atol=atol),
                     point_on_line([x1, y1], line1, rtol=rtol, atol=atol),
                     point_on_line([x2, y2], line0, rtol=rtol, atol=atol),
                     point_on_line([x3, y3], line0, rtol=rtol, atol=atol))

            return collinearmap[state]([x0, y0], [x1, y1],
                                       [x2, y2], [x3, y3])
        else:
            # Lines are parallel but aren't collinear
            return 4, None
    else:
        # Lines are not parallel, check if they intersect
        u0 = u0 / denom
        u1 = u1 / denom

        x = x0 + u0 * (x1 - x0)
        y = y0 + u0 * (y1 - y0)

        # Sanity check - can be removed to speed up if needed
        if not np.allclose(x, x2 + u1 * (x3 - x2), rtol=rtol, atol=atol):
            raise Exception
        if not np.allclose(y, y2 + u1 * (y3 - y2), rtol=rtol, atol=atol):
            raise Exception

        # Check if point found lies within given line segments
        if 0.0 <= u0 <= 1.0 and 0.0 <= u1 <= 1.0:
            # We have intersection
            return 1, np.array([x, y])
        else:
            # No intersection
            return 0, None
