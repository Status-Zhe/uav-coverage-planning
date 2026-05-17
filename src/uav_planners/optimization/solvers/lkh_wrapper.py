"""LKH (Lin-Kernighan-Helsgaun) TSP solver wrapper.

LKH is a powerful heuristic for solving large-scale traveling salesman problems.
This wrapper provides a Python interface to the LKH solver.

Note: LKH binary must be installed separately. This wrapper assumes the 'lkh'
command is available in the system PATH.
"""

import os
import subprocess
import tempfile
from typing import List, Optional
import numpy as np

from ...models.waypoint import Waypoint


class LKHWrapper:
    """Wrapper for LKH TSP solver.
    
    LKH is one of the best heuristics for large-scale TSP instances.
    It can handle problems with thousands of nodes efficiently.
    
    Requirements:
        - LKH binary installed and available in PATH
        
    Installation:
        http://webhotel4.ruc.dk/~keld/research/LKH/
    
    Example:
        lkh = LKHWrapper()
        ordered_waypoints = lkh.solve(waypoints)
    """
    
    def __init__(self, lkh_path: str = "lkh", timeout: int = 60):
        """Initialize LKH wrapper.
        
        Args:
            lkh_path: Path to LKH binary (default: "lkh")
            timeout: Maximum solving time in seconds
        """
        self.lkh_path = lkh_path
        self.timeout = timeout
        self._check_lkh_available()
    
    def _check_lkh_available(self) -> bool:
        """Check if LKH is available.
        
        Returns:
            True if LKH is available
            
        Raises:
            RuntimeError: If LKH is not found
        """
        try:
            result = subprocess.run(
                [self.lkh_path, "-h"],
                capture_output=True,
                timeout=5
            )
            return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    @property
    def is_available(self) -> bool:
        """Check if LKH solver is available.
        
        Returns:
            True if LKH is installed and available
        """
        return self._check_lkh_available()
    
    def solve(self, waypoints: List[Waypoint]) -> Optional[List[Waypoint]]:
        """Solve TSP using LKH.
        
        Args:
            waypoints: Unordered list of waypoints
            
        Returns:
            Ordered list of waypoints, or None if LKH not available
        """
        if not self.is_available:
            return None
        
        if len(waypoints) <= 1:
            return waypoints.copy()
        
        # Compute distance matrix
        distance_matrix = self._compute_distance_matrix(waypoints)
        
        # Create LKH input files
        with tempfile.TemporaryDirectory() as tmpdir:
            par_file = os.path.join(tmpdir, "input.par")
            tsp_file = os.path.join(tmpdir, "input.tsp")
            tour_file = os.path.join(tmpdir, "output.tour")
            
            # Write TSP file
            self._write_tsp_file(tsp_file, distance_matrix)
            
            # Write parameter file
            self._write_par_file(par_file, tsp_file, tour_file)
            
            # Run LKH
            try:
                result = subprocess.run(
                    [self.lkh_path, par_file],
                    capture_output=True,
                    timeout=self.timeout,
                    cwd=tmpdir
                )
                
                if result.returncode != 0:
                    return None
                
                # Read solution
                tour = self._read_tour_file(tour_file)
                
                if tour is None or len(tour) != len(waypoints):
                    return None
                
                # Reorder waypoints according to tour
                ordered = [waypoints[i] for i in tour]
                return ordered
                
            except subprocess.TimeoutExpired:
                return None
            except Exception:
                return None
    
    def _compute_distance_matrix(self, waypoints: List[Waypoint]) -> np.ndarray:
        """Compute distance matrix between all waypoints.
        
        Args:
            waypoints: List of waypoints
            
        Returns:
            NxN distance matrix
        """
        n = len(waypoints)
        coords = np.array([[wp.x, wp.y, wp.z] for wp in waypoints])
        
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff ** 2, axis=2))
        
        # Scale to integers (LKH requires integers)
        distances = (distances * 1000).astype(int)
        
        return distances
    
    def _write_tsp_file(self, filepath: str, distance_matrix: np.ndarray) -> None:
        """Write TSP problem file in TSPLIB format.
        
        Args:
            filepath: Output file path
            distance_matrix: Distance matrix
        """
        n = len(distance_matrix)
        
        with open(filepath, 'w') as f:
            f.write("NAME: UAV_TSP\n")
            f.write("TYPE: TSP\n")
            f.write(f"DIMENSION: {n}\n")
            f.write("EDGE_WEIGHT_TYPE: EXPLICIT\n")
            f.write("EDGE_WEIGHT_FORMAT: FULL_MATRIX\n")
            f.write("EDGE_WEIGHT_SECTION\n")
            
            for i in range(n):
                row = ' '.join(str(distance_matrix[i, j]) for j in range(n))
                f.write(row + "\n")
            
            f.write("EOF\n")
    
    def _write_par_file(
        self,
        par_file: str,
        tsp_file: str,
        tour_file: str
    ) -> None:
        """Write LKH parameter file.
        
        Args:
            par_file: Parameter file path
            tsp_file: TSP file path
            tour_file: Output tour file path
        """
        with open(par_file, 'w') as f:
            f.write(f"PROBLEM_FILE = {tsp_file}\n")
            f.write(f"TOUR_FILE = {tour_file}\n")
            f.write("RUNS = 1\n")
            f.write("TRACE_LEVEL = 0\n")
    
    def _read_tour_file(self, filepath: str) -> Optional[List[int]]:
        """Read solution tour from LKH output.
        
        Args:
            filepath: Tour file path
            
        Returns:
            List of node indices, or None if error
        """
        if not os.path.exists(filepath):
            return None
        
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            tour = []
            in_tour = False
            
            for line in lines:
                line = line.strip()
                
                if line == "TOUR_SECTION":
                    in_tour = True
                    continue
                
                if line == "-1" or line == "EOF":
                    break
                
                if in_tour and line:
                    try:
                        node = int(line) - 1  # LKH uses 1-based indexing
                        if node >= 0:
                            tour.append(node)
                    except ValueError:
                        continue
            
            return tour if tour else None
            
        except Exception:
            return None
