from __future__ import print_function,division
from builtins import range
from six import iteritems

import math
import random
from . import differences
from . import sampling
from ..klampt import vectorops
import numpy as np
from OpenGL.GL import *


class Set:
    """Abstract base class for a set in a d-dimensional vector space. 
    
    A set, at the minimum, has an inside-outside test.  It may be
    optionally bounded and sample-able.
    """
    def __str__(self):
        return self.__class__.__name__
        
    def dimension(self):
        """Returns the number of entries of an element of this set"""
        try:
            x = self.sample()
            return len(x)
        except NotImplementedError:
            raise NotImplementedError("Set "+str(self)+" does not implement dimension()")
            
    def bounds(self):
        """Returns a pair (bmin,bmax) giving an axis-aligned bounding box
        on items in this set.  Return None if no bound can be determined."""
        raise NotImplementedError("Set "+str(self)+" does not implement bounds()")
        
    def sample(self):
        """Sample a random value from the set"""
        raise NotImplementedError("Set "+str(self)+" does not implement sample()")
        
    def contains(self,x):
        """Returns True if x is in the set."""
        raise NotImplementedError("Set "+str(self)+" does not implement contains()")
    
    def project(self,x):
        """If x is not contained in the set, returns a nearby point in the
        set.  If is is contained, just returns x."""
        raise NotImplementedError("Set "+str(self)+" does not implement project()")

    def signedDistance(self,x):
        """Returns a signed distance function d(x) that is > 0 when x is
        outside of the set, and < 0 when x is inside.  d(x)=0 if x is on
        the boundary.
        
        Required for numerical optimization methods."""
        raise NotImplementedError("Set "+str(self)+" does not implement signedDistance()")
    
    def signedDistance_gradient(self,x):
        """Required for numerical optimization methods"""
        return differences.gradient_forward_difference(self.signedDistance,x,1e-4)


class SingletonSet(Set):
    """A single point {x}"""
    def __init__(self,x):
        self.x = x

    def bounds(self):
        return [self.x,self.x]

    def contains(self,x):
        return x == self.x

    def sample(self):
        return self.x

    def project(self,x):
        return self.x

    def signedDistance(self,x):
        d = np.asarray(x)-np.asarray(self.x)
        return np.dot(d,d)
    
    def signedDistance_gradient(self,x):
        d = np.asarray(x)-np.asarray(self.x)
        return 2*d


class NeighborhoodSet(Set):
    """A ball of radius r around a point c"""
    def __init__(self,c,r):
        self.c = c
        self.r = r
        
    def bounds(self):
        return [[v-self.r for v in self.c],[v+self.r for v in self.c]]

    def contains(self,x):
        return vectorops.distance(x,self.c) <= self.r

    def sample(self):
        return sampling.sample_hyperball(len(self.c),self.c,self.r)

    def project(self,x):
        d = vectorops.distance(x,self.c)
        if d <= self.r: return x
        return vectorops.madd(x,vectorops.sub(self.c,x),(d-self.r)/d)

    def signedDistance(self,x):
        d = vectorops.distance(x,self.c)
        return d - self.r


class FiniteSet(Set):
    """Represents a finite set of objects in a vector space."""
    def __init__(self,items,metric=None):
        self.items = items
        if metric is None:
            self.metric = vectorops.distance
        else:
            self.metric = metric
    def bounds(self):
        bmin = self.items[0][:]
        bmax = self.items[0][:]
        for v in self.items[1:]:
            for i in range(len(v)):
                if v[i] < bmin[i]: bmin[i] = v[i]
                elif v[i] > bmax[i]: bmax[i] = v[i]
        return (bmin,bmax)
    def sample(self):
        return random.choice(self.items)
    def contains(self,x):
        return x in self.items
    def project(self,x):
        if len(self.items)==0:
            return None
        (d,pt) = sorted([(self.metric(x,p),p) for p in self.items])[0]
        return pt
    def signedDistance(self,x):
        if self.metric is not vectorops.distance:
            print("WARNING: FiniteSet metric is not euclidean distance, treating as Euclidean in signedDistance")
        mind = float('inf')
        x = np.asarray(x)
        for v in self.items:
            v = np.asarray(v)
            d = self.metric(x,v)
            mind = min(d,mind)
        return mind
    def signedDistance_gradient(self,x):
        if self.metric is not vectorops.distance:
            print("WARNING: FiniteSet metric is not euclidean distance, treating as Euclidean in signedDistance")
        mind = float('inf')
        minv = None
        x = np.asarray(x)
        for v in self.items:
            v = np.asarray(v)
            d = np.dot(x-v,x-v)
            if d < mind:
                minv = v
            mind = min(d,mind)
        if minv is None:
            return np.zeros(len(x))
        return 2*(x-minv)


class BoxPivotNonCaptureSet(Set):
    """Represents a non-maneuverable set of state that the pivoting point has a high velocity."""
    def __init__(self,bmin,bmax):
        self.bmin = bmin
        self.bmax = bmax
    def dimension(self):
        return len(self.bmin)
    def bounds(self):
        return (self.bmin,self.bmax)
    def sample(self):
        pass
    def contains(self, x, vel_thres=1.5):
        assert len(x)==len(self.bmin)
        point_relative_position = [2, 0, -2]
        angular_velocity = [0, x[5], 0]
        linear_velocity = [x[3], 0, x[4]]
        point_velocity = linear_velocity + np.cross(angular_velocity, point_relative_position)
        point_velocity = np.linalg.norm(point_velocity)
        if point_velocity > vel_thres:
            return True
        else:
            return False
    

class GripperNonCaptureSet(Set):
    """Represents a non-capture set of state that the box is not inside the hand."""
    def __init__(self, bmin, bmax, success_z_thres, success_vz_thres):
        self.bmin = bmin
        self.bmax = bmax
        self.success_z_thres = success_z_thres
        self.success_vz_thres = success_vz_thres
    def dimension(self):
        return len(self.bmin)
    def bounds(self):
        return (self.bmin,self.bmax)
    def sample(self):
        pass
    def contains(self, x):
        if x[8] < self.success_vz_thres or x[1] < self.success_z_thres: # high velocity or under the hand
            return True # non-capture
        else:
            return False
    

class GripperSuccessSet(Set):
    """Represents a success set of state."""
    def __init__(self, bmin, bmax, success_z_thres, success_vz_thres):
        self.bmin = bmin
        self.bmax = bmax
        self.success_z_thres = success_z_thres
        self.success_vz_thres = success_vz_thres
    def dimension(self):
        return len(self.bmin)
    def bounds(self):
        return (self.bmin,self.bmax)
    def sample(self):
        pass
    def contains(self, x):
        if x[8] > self.success_vz_thres and x[1] > self.success_z_thres: # low velocity and inside the hand
            return True # successfully grasped
        else:
            return False


class GripperMultiCaptureSet(Set):
    """Represents a capture set of state."""
    def __init__(self, bmin, bmax, z_thres, vz_thres, num_objects):
        self.bmin = bmin
        self.bmax = bmax
        self.z_thres = z_thres
        self.vz_thres = vz_thres
        self.num_objects = num_objects
    def dimension(self):
        return len(self.bmin)
    def bounds(self):
        return (self.bmin, self.bmax)
    def sample(self):
        pass
    def contains(self, x):
        if all((x[2+12*i] > self.z_thres and x[8+12*i] > self.vz_thres) for i in range(self.num_objects)):
            return True # capture indicated by the z-position over tabletop and small absolute z-velocity
        else:
            return False


class GripperMultiComplementCaptureSet(Set):
    """Represents a capture set of state."""
    def __init__(self, bmin, bmax, z_thres, vz_thres, num_objects):
        self.bmin = bmin
        self.bmax = bmax
        self.z_thres = z_thres
        self.vz_thres = vz_thres
        self.num_objects = num_objects
    def dimension(self):
        return len(self.bmin)
    def bounds(self):
        return (self.bmin, self.bmax)
    def sample(self):
        pass
    def contains(self, x):
        if all((x[2+12*i] < self.z_thres or x[8+12*i] < self.vz_thres) for i in range(self.num_objects)):
            return True # non-capture indicated by the z-position on/below tabletop or large absolute z-velocity
        else:
            return False


class GripperMultiSuccessSet(Set):
    """Represents a success set of state."""
    def __init__(self, bmin, bmax, success_z_thres, num_objects):
        self.bmin = bmin
        self.bmax = bmax
        self.success_z_thres = success_z_thres
        # self.success_vz_thres = success_vz_thres
        self.num_objects = num_objects
    def dimension(self):
        return len(self.bmin)
    def bounds(self):
        return (self.bmin, self.bmax)
    def sample(self):
        pass
    def contains(self, x):
        if all(x[2+12*i] > self.success_z_thres for i in range(self.num_objects)): # above a level
            return True # successfully lifted above a horizontal plane
        else:
            return False


class GripperMultiComplementSuccessSet(Set):
    """Represents a non-success set of state."""
    def __init__(self, bmin, bmax, success_z_thres, num_objects):
        self.bmin = bmin
        self.bmax = bmax
        self.success_z_thres = success_z_thres
        self.num_objects = num_objects
    def dimension(self):
        return len(self.bmin)
    def bounds(self):
        return (self.bmin, self.bmax)
    def sample(self):
        pass
    def contains(self, x):
        if all(x[2+12*i] < self.success_z_thres for i in range(self.num_objects)): # above a level
            return True # successfully lifted above a horizontal plane
        else:
            return False
        

class BoxSet(Set):
    """Represents an axis-aligned box in a vector space."""
    def __init__(self,bmin,bmax):
        self.bmin = bmin
        self.bmax = bmax
    def dimension(self):
        return len(self.bmin)
    def bounds(self):
        return (self.bmin,self.bmax)
    def sample(self):
        return [random.uniform(a,b) for (a,b) in zip(self.bmin,self.bmax)]
    def contains(self,x):
        assert len(x)==len(self.bmin)
        for (xi,a,b) in zip(x,self.bmin,self.bmax):
            if xi < a or xi > b:
                return False
        return True
    def project(self,x):
        assert len(x)==len(self.bmin)
        assert len(x)==len(self.bmax)
        xnew = x[:]
        for i,(xi,a,b) in enumerate(zip(x,self.bmin,self.bmax)):
            if xi < a:
                xnew[i] = a
            elif xi > b:
                xnew[i] = b
        return xnew
    def signedDistance(self,x):
        xclamp = np.zeros(len(x))
        assert len(x)==len(self.bmin)
        mindist = float('inf')
        for i,(xi,a,b) in enumerate(zip(x,self.bmin,self.bmax)):
            xclamp[i] = min(b,max(xi,a))
            mindist = min(mindist,xi-a,b-xi)
        if mindist < 0:
            #outside
            x = np.asarray(x)
            return np.dot(x-xclamp,x-xclamp)
        else:
            #inside
            return -mindist
    def signedDistance_gradient(self,x):
        xclamp = np.empty(len(x))
        assert len(x)==len(self.bmin)
        mindist = float('inf')
        imindist = None
        iminsign = 1.0
        for i,(xi,a,b) in enumerate(zip(x,self.bmin,self.bmax)):
            xclamp[i] = min(b,max(xi,a))
            if xi-a < mindist:
                imindist = i
                iminsign = -1.0
                mindist = xi-a
            if b-xi < mindist:
                imindist = i
                iminsign = 1.0
                mindist = b-xi
        if mindist < 0:
            #outside
            x = np.asarray(x)
            return 2*(x-xclamp)
        else:
            #inside
            res = np.zeros(len(x))
            res[imindist] = iminsign
            return res

    def drawGL(self, c=[1,0,0,0.2]):
        # Enable blending for transparency
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Set color with alpha for transparency (RGBA)
        glColor4f(*c)  # White color with 50% transparency

        glBegin(GL_QUADS)
        glVertex2f(*self.bmin[:2])
        glVertex2f(self.bmax[0],self.bmin[1])
        glVertex2f(*self.bmax[:2])
        glVertex2f(self.bmin[0],self.bmax[1])
        glEnd()

class UnionBoxSet(Set):
    """Represents the union of several axis-aligned box sets in a vector space."""
    def __init__(self, bmin, bmax):
        """Initialize the UnionBoxSet with lists of minimum and maximum bounds for each box.
        
        Args:
            bmin (list): A list of lists of minimum bounds for each box.
            bmax (list): A list of lists of maximum bounds for each box.
        """
        self.bmin = bmin
        self.bmax = bmax
        assert len(bmin) == len(bmax), "bmin and bmax must be lists of the same length."
        self.boxes = [BoxSet(bmin[i], bmax[i]) for i in range(len(bmin))]

    def dimension(self):
        return self.boxes[0].dimension() if self.boxes else 0

    def bounds(self):
        all_bmin, all_bmax = zip(*(box.bounds() for box in self.boxes))
        bmin = [min(bounds) for bounds in zip(*all_bmin)]
        bmax = [max(bounds) for bounds in zip(*all_bmax)]
        return bmin, bmax

    def contains(self, x):
        return any(box.contains(x) for box in self.boxes)

    def sample(self):
        box = random.choice(self.boxes)
        return box.sample()

    def project(self, x):
        # Project onto the box that x is closest to (in terms of the box center)
        closest_box = min(self.boxes, key=lambda box: vectorops.distance(x, vectorops.div(vectorops.add(box.bmin, box.bmax), 2)))
        return closest_box.project(x)

    def signedDistance(self, x):
        # Return the smallest signed distance among all boxes
        return min(box.signedDistance(x) for box in self.boxes)

    def signedDistance_gradient(self, x):
        # Calculate the gradient for the box with the smallest signed distance
        closest_box = min(self.boxes, key=lambda box: box.signedDistance(x))
        return closest_box.signedDistance_gradient(x)
    
    def drawGL(self, c=[1,0,0,0.2]):
        for box in self.boxes:
            box.drawGL(c)
    

class RingSet(Set):
    """
    Represents a ring in 2D space, defined as the region between two circles with radii r1 and r2.
    """
    def __init__(self, c, r1, r2):
        """
        Initializes the RingSet.
        :param c: Center of the ring as a 2D tuple or list (x, y).
        :param r1: Inner radius of the ring.
        :param r2: Outer radius of the ring.
        """
        assert r1 < r2, "Inner radius must be smaller than outer radius."
        self.c = np.array(c)
        self.r1 = r1
        self.r2 = r2

    def bounds(self):
        """
        Returns the bounding box of the ring.
        """
        return ([self.c[0] - self.r2, self.c[1] - self.r2],
                [self.c[0] + self.r2, self.c[1] + self.r2])

    def contains(self, x):
        """
        Checks if a point is inside the ring.
        """
        dist_sq = np.sum((np.array(x) - self.c) ** 2)
        return self.r1**2 <= dist_sq <= self.r2**2

    def sample(self):
        """
        Samples a random point from the ring.
        """
        angle = random.uniform(0, 2 * math.pi)
        radius = math.sqrt(random.uniform(self.r1**2, self.r2**2))
        return [self.c[0] + radius * math.cos(angle), self.c[1] + radius * math.sin(angle)]

    def project(self, x):
        """
        Projects a point onto the nearest point in the ring.
        """
        x_vec = np.array(x) - self.c
        dist_sq = np.sum(x_vec ** 2)
        if dist_sq < self.r1**2:
            return self.c + x_vec * (self.r1 / np.sqrt(dist_sq))
        elif dist_sq > self.r2**2:
            return self.c + x_vec * (self.r2 / np.sqrt(dist_sq))
        return x

    def signedDistance(self, x):
        """
        Returns the signed distance of x from the boundary of the ring.
        """
        dist = np.sqrt(np.sum((np.array(x) - self.c) ** 2))
        if dist <= self.r1:
            return self.r1 - dist
        elif dist >= self.r2:
            return dist - self.r2
        return min(dist - self.r1, self.r2 - dist)

    def signedDistance_gradient(self, x):
        """
        Returns the gradient of the signed distance function at x.
        """
        x_vec = np.array(x) - self.c
        dist = np.sqrt(np.sum(x_vec ** 2))
        if dist == 0:
            return np.zeros_like(x)
        grad_outside = x_vec / dist
        if dist <= self.r1:
            return -grad_outside
        elif dist >= self.r2:
            return grad_outside
        # If x is inside the ring, the gradient points towards the nearest boundary.
        if dist - self.r1 < self.r2 - dist:
            return -grad_outside
        return grad_outside
    
    def drawGL(self, color=[1,0,0,0.2]):
        # Enable blending for transparency
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Set color with alpha for transparency (RGBA)
        glColor4f(*color)

        # Draw the outer circle
        glBegin(GL_TRIANGLE_FAN)
        glVertex2f(*self.c)
        for i in range(101):
            angle = 2 * math.pi * i / 100
            glVertex2f(self.c[0] + self.r2 * math.cos(angle), self.c[1] + self.r2 * math.sin(angle))
        glEnd()        
        
        # Draw the inner circle in white
        glColor4f(1, 1, 1, 1)
        glBegin(GL_TRIANGLE_FAN)
        for i in range(101):
            angle = 2 * math.pi * i / 100
            glVertex2f(self.c[0] + self.r1 * math.cos(angle), self.c[1] + self.r1 * math.sin(angle))
        glEnd()        


class CaptureSetClass(Set):
    """Represents the shape remaining on a square canvas after erasing along an arc trajectory."""
    def __init__(self, canvas_limits, eraser_radius, arc_center, arc_radius, arc_angle_range):
        """
        Initializes the CaptureSetClass.
        
        :param canvas_limits: Boundaries of the square canvas [bmin, bmax], bmin=[bmin_d1, ...,bmin_dn].
        :param eraser_radius: Radius of the circular eraser.
        :param arc_center: Center of the arc (x, y).
        :param arc_radius: Radius of the arc.
        :param arc_angle_range: Angle range of the arc (start_angle, end_angle) in radians.
        """
        self.canvas_limits = canvas_limits
        self.eraser_radius = eraser_radius
        self.arc_center = np.array(arc_center)
        self.arc_radius = arc_radius
        self.arc_angle_range = arc_angle_range

    def dimension(self):
        return len(self.canvas_limits[0])
    
    def contains(self, x):
        """
        Check if a point x is in the remaining red area.
        """
        # Check if point is inside the canvas
        if not (self.canvas_limits[0][0] <= x[0] <= self.canvas_limits[1][0] and self.canvas_limits[0][1] <= x[1] <= self.canvas_limits[1][1]):
            return False

        # Check if point is outside the erased arc path
        dist_to_arc_center = np.linalg.norm(np.array(x) - self.arc_center)
        angle_to_point = np.arctan2(x[1] - self.arc_center[1], x[0] - self.arc_center[0])

        # Normalize the angle_to_point within the range [0, 2π)
        angle_to_point = (angle_to_point + 2 * np.pi) % (2 * np.pi)

        # Check if the point is within the angle range and the eraser's effect
        if self.arc_angle_range[0] <= self.arc_angle_range[1]:
            angle_in_range = self.arc_angle_range[0] <= angle_to_point <= self.arc_angle_range[1]
        else: # the range comes over 2pi
            angle_in_range = self.arc_angle_range[0] <= angle_to_point or angle_to_point <= self.arc_angle_range[1]
            
        if (angle_in_range and
            self.arc_radius - self.eraser_radius <= dist_to_arc_center <= self.arc_radius + self.eraser_radius):
            return True

        return False

    def drawGL(self, color=[1,0,0,0.2], res=0.01):
        # Draw the square canvas
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glColor4f(*color)
        glBegin(GL_QUADS)
        glVertex2f(self.canvas_limits[0][0], self.canvas_limits[0][1])
        glVertex2f(self.canvas_limits[1][0], self.canvas_limits[0][1])
        glVertex2f(self.canvas_limits[1][0], self.canvas_limits[1][1])
        glVertex2f(self.canvas_limits[0][0], self.canvas_limits[1][1])
        glEnd()

        # Draw the erased arc region
        glColor3f(212/255.0, 241/255.0, 255/255)  # blue color to represent the erased area
        if self.arc_angle_range[0] <= self.arc_angle_range[1]:
            self.draw_triangle_strips(self.arc_angle_range[0], self.arc_angle_range[1], res)
        else:
            range_list = [[self.arc_angle_range[0], 2*np.pi-1e-9], [0.0, self.arc_angle_range[1]]]
            for i in range(len(range_list)):
                self.draw_triangle_strips(range_list[i][0], range_list[i][1], res)
                
    def draw_triangle_strips(self, lowb, upb, res=0.01):
        numdivs = int(math.ceil((upb - lowb) * self.arc_radius / res))
        glBegin(GL_TRIANGLE_STRIP)
        for i in range(numdivs + 1):
            u = lowb + float(i) / float(numdivs) * (upb - lowb)
            inner_x = self.arc_center[0] + (self.arc_radius - self.eraser_radius) * math.cos(u)
            inner_y = self.arc_center[1] + (self.arc_radius - self.eraser_radius) * math.sin(u)
            outer_x = self.arc_center[0] + (self.arc_radius + self.eraser_radius) * math.cos(u)
            outer_y = self.arc_center[1] + (self.arc_radius + self.eraser_radius) * math.sin(u)
            glVertex2f(inner_x, inner_y)
            glVertex2f(outer_x, outer_y)
        glEnd()


class ComplementCaptureSetClass(Set):
    """Represents the shape remaining on a square canvas after erasing along an arc trajectory."""
    def __init__(self, canvas_limits, eraser_radius, arc_center, arc_radius, arc_angle_range):
        """
        Initializes the ComplementCaptureSetClass.
        
        :param canvas_limits: Boundaries of the square canvas [bmin, bmax], bmin=[bmin_d1, ...,bmin_dn].
        :param eraser_radius: Radius of the circular eraser.
        :param arc_center: Center of the arc (x, y).
        :param arc_radius: Radius of the arc.
        :param arc_angle_range: Angle range of the arc (start_angle, end_angle) in radians.
        """
        self.canvas_limits = canvas_limits
        self.eraser_radius = eraser_radius
        self.arc_center = np.array(arc_center)
        self.arc_radius = arc_radius
        self.arc_angle_range = arc_angle_range

    def dimension(self):
        return len(self.canvas_limits[0])
    
    def contains(self, x):
        """
        Check if a point x is in the remaining red area.
        """
        # Check if point is inside the canvas
        if not (self.canvas_limits[0][0] <= x[0] <= self.canvas_limits[1][0] and self.canvas_limits[0][1] <= x[1] <= self.canvas_limits[1][1]):
            return False

        # Check if point is outside the erased arc path
        dist_to_arc_center = np.linalg.norm(np.array(x) - self.arc_center)
        angle_to_point = np.arctan2(x[1] - self.arc_center[1], x[0] - self.arc_center[0])

        # Normalize the angle_to_point within the range [0, 2π)
        angle_to_point = (angle_to_point + 2 * np.pi) % (2 * np.pi)

        # Check if the point is within the angle range and the eraser's effect
        if self.arc_angle_range[0] <= self.arc_angle_range[1]:
            angle_in_range = self.arc_angle_range[0] <= angle_to_point <= self.arc_angle_range[1]
        else: # the range comes over 2pi
            angle_in_range = self.arc_angle_range[0] <= angle_to_point or angle_to_point <= self.arc_angle_range[1]
            
        if (angle_in_range and
            self.arc_radius - self.eraser_radius <= dist_to_arc_center <= self.arc_radius + self.eraser_radius):
            return False

        return True

    def sample(self):
        while True:
            # Randomly sample a point within the canvas
            x = random.uniform(self.canvas_limits[0][0], self.canvas_limits[1][0])
            y = random.uniform(self.canvas_limits[0][1], self.canvas_limits[1][1])
            point = [x, y]

            # Check if the point is not in the erased area
            if self.contains(point):
                return point
                
    def draw_triangle_strips(self, lowb, upb, res=0.01):
        numdivs = int(math.ceil((upb - lowb) * self.arc_radius / res))
        glBegin(GL_TRIANGLE_STRIP)
        for i in range(numdivs + 1):
            u = lowb + float(i) / float(numdivs) * (upb - lowb)
            inner_x = self.arc_center[0] + (self.arc_radius - self.eraser_radius) * math.cos(u)
            inner_y = self.arc_center[1] + (self.arc_radius - self.eraser_radius) * math.sin(u)
            outer_x = self.arc_center[0] + (self.arc_radius + self.eraser_radius) * math.cos(u)
            outer_y = self.arc_center[1] + (self.arc_radius + self.eraser_radius) * math.sin(u)
            glVertex2f(inner_x, inner_y)
            glVertex2f(outer_x, outer_y)
        glEnd()

    def drawGL(self, color=[1,0,0,0.2], res=0.01):
        # Draw the square canvas
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glColor4f(*color)
        glBegin(GL_QUADS)
        glVertex2f(self.canvas_limits[0][0], self.canvas_limits[0][1])
        glVertex2f(self.canvas_limits[1][0], self.canvas_limits[0][1])
        glVertex2f(self.canvas_limits[1][0], self.canvas_limits[1][1])
        glVertex2f(self.canvas_limits[0][0], self.canvas_limits[1][1])
        glEnd()

        # Draw the erased arc region
        glColor3f(212/255.0, 241/255.0, 255/255)  # blue color to represent the erased area
        if self.arc_angle_range[0] <= self.arc_angle_range[1]:
            self.draw_triangle_strips(self.arc_angle_range[0], self.arc_angle_range[1], res)
        else:
            range_list = [[self.arc_angle_range[0], 2*np.pi-1e-9], [0.0, self.arc_angle_range[1]]]
            for i in range(len(range_list)):
                self.draw_triangle_strips(range_list[i][0], range_list[i][1], res)

class LambdaSet(Set):
    """Given some standalone function fcontains(x) which determines
    membership, produces a Set object.

    Optionally, a function fsample() can be provided to sample from the
    set."""
    def __init__(self,fcontains,fsample=None):
        self.fcontains = fcontains
        self.fsample = fsample
    def sample(self):
        if self.fsample:
            return self.fsample()
        return None
    def contains(self,x):
        return self.fcontains(x)


class MultiSet(Set):
    """A cartesian product of sets"""
    def __init__(self,*components):
        self.components = components
    def __str__(self):
        return ' x '.join(str(c) for c in self.components)
    def dimension(self):
        return sum(c.dimension() for c in self.components)
    def bounds(self):
        cbounds = [c.bounds() for c in self.components]
        if any(c==None for c in cbounds): return None
        bmin,bmax = zip(*cbounds)
        return self.join(bmin),self.join(bmax)
    def split(self,x):
        i = 0
        res = []
        for c in self.components:
            d = c.dimension()
            res.append(x[i:i+d])
            i += d
        return res
    def join(self,xs):
        return sum(xs,[])
    def contains(self,x):
        for (c,xi) in zip(self.components,self.split(x)):
            if not c.contains(xi):
                return False
        return True
    def sample(self):
        return self.join(c.sample() for c in self.components)
    def project(self,x):
        return self.join(c.project(xi) for c,xi in zip(self.components,self.split(x)))
    def signedDistance(self,x):
        return max(c.signedDistance(xi) for (c,xi) in zip(self.components,self.split(x)))
    def signedDistance_gradient(self,x):
        xis = self.split(x)
        pieces = [np.zeros(len(xi)) for xi in xis]
        imin = max((c.signedDistance(xi),i) for i,(c,xi) in enumerate(zip(self.components,xis)))[1]
        pieces[imin] = self.components[imin].signedDistance_gradient(xis[imin])
        return np.hstack(pieces)
