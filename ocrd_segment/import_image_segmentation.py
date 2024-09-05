from __future__ import absolute_import

from typing import Optional
from PIL import Image

import numpy as np
import cv2

from ocrd_utils import (
    points_from_polygon,
    pushd_popd,
    membername
)
# pragma pylint: disable=unused-import
# (region types will be referenced indirectly via globals())
from ocrd_models.ocrd_page import (
    OcrdPage,
    CoordsType,
    TextRegionType,
    ImageRegionType,
    MathsRegionType,
    SeparatorRegionType,
    NoiseRegionType,
)
from ocrd_models.ocrd_page_generateds import (
    BorderType,
    TableRegionType,
    GraphicRegionType,
    ChartRegionType,
    ChemRegionType,
    LineDrawingRegionType,
    MusicRegionType,
    UnknownRegionType,
    TextTypeSimpleType,
    GraphicsTypeSimpleType,
    ChartTypeSimpleType
)
# pragma pylint: enable=unused-import
from ocrd import Processor, OcrdPageResult

TYPEDICT = {
    "TextRegion": TextTypeSimpleType,
    "GraphicRegion": GraphicsTypeSimpleType,
    "ChartType": ChartTypeSimpleType
}


class ImportImageSegmentation(Processor):

    @property
    def executable(self):
        return 'ocrd-segment-from-masks'

    def process_page_pcgts(self, *input_pcgts: Optional[OcrdPage], page_id: Optional[str] = None) -> OcrdPageResult:
        """Performs region segmentation by reading mask images in pseudo-colour.

        Open and deserialize PAGE input file (or generate from image input file)
        from the first input file group, as well as mask image file from the second.

        Then iterate over all connected (equally colored) mask segments and compute
        convex hull contours for them. Convert them to polygons, and look up their
        color value in ``colordict`` to instantiate the appropriate region types
        (optionally with subtype). Instantiate and annotate regions accordingly.

        Produce a new output file by serialising the resulting hierarchy.
        """
        colordict = self.parameter['colordict']

        pcgts = input_pcgts[0]
        page = pcgts.get_Page()

        # import mask image (for which Processor.process_page_file will have created a pseudo PAGE by now)
        segmentation_filename = input_pcgts[1].get_Page().get_imageFilename()
        segmentation_pil = Image.open(segmentation_filename)
        has_alpha = segmentation_pil.mode == 'RGBA'
        if has_alpha:
            colorformat = "%08X"
        else:
            colorformat = "%06X"
            if segmentation_pil.mode != 'RGB':
                segmentation_pil = segmentation_pil.convert('RGB')
        # convert to array
        segmentation_array = np.array(segmentation_pil)
        # collapse 3 color channels
        segmentation_array = segmentation_array.dot(
            np.array([2**24, 2**16, 2**8, 1], np.uint32)[0 if has_alpha else 1:])
        # partition mapped colors vs background
        colors = np.unique(segmentation_array)
        bgcolors = []
        for i, color in enumerate(colors):
            colorname = colorformat % color
            if (colorname not in colordict or
                not colordict[colorname]):
                #raise Exception("Unknown color %s (not in colordict)" % colorname)
                self.logger.info("Ignoring background color %s", colorname)
                bgcolors.append(i)
        background = np.zeros_like(segmentation_array, np.uint8)
        if bgcolors:
            for i in bgcolors:
                background += np.array(segmentation_array == colors[i], np.uint8)
            colors = np.delete(colors, bgcolors, 0)
        # iterate over mask for each mapped color/class
        regionno = 0
        for color in colors:
            # get region (sub)type
            colorname = colorformat % color
            classname = colordict[colorname]
            regiontype = None
            custom = None
            if ":" in classname:
                classname, regiontype = classname.split(":")
                if classname in TYPEDICT:
                    typename = membername(TYPEDICT[classname], regiontype)
                    if typename == regiontype:
                        # not predefined in PAGE: use other + custom
                        custom = "subtype:%s" % regiontype
                        regiontype = "other"
                else:
                    custom = "subtype:%s" % regiontype
            if classname + "Type" not in globals():
                raise Exception("Unknown class '%s' for color %s in colordict" % (classname, colorname))
            classtype = globals()[classname + "Type"]
            if classtype is BorderType:
                # mask from all non-background regions
                classmask = 1 - background
            else:
                # mask from current color/class
                classmask = np.array(segmentation_array == color, np.uint8)
            if not np.count_nonzero(classmask):
                continue
            # now get the contours and make polygons for them
            contours, _ = cv2.findContours(classmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                # (could also just take bounding boxes to avoid islands/inclusions...)
                area = cv2.contourArea(contour)
                # filter too small regions
                area_pct = area / np.prod(segmentation_array.shape) * 100
                if area < 100 and area_pct < 0.1:
                    self.logger.warning('ignoring contour of only %.1f%% area for %s',
                                        area_pct, classname)
                    continue
                self.logger.info('found region %s:%s:%s with area %.1f%%',
                                 classname, regiontype or '', custom or '', area_pct)
                # simplify shape
                poly = cv2.approxPolyDP(contour, 2, False)[:, 0, ::] # already ordered x,y
                if len(poly) < 4:
                    self.logger.warning('ignoring contour of only %d points (area %.1f%%) for %s',
                                        len(poly), area_pct, classname)
                    continue
                if classtype is BorderType:
                    # add Border
                    page.set_Border(BorderType(Coords=CoordsType(points=points_from_polygon(poly))))
                    break
                else:
                    # instantiate region
                    regionno += 1
                    region = classtype(id="region_%d" % regionno, type_=regiontype, custom=custom,
                                       Coords=CoordsType(points=points_from_polygon(poly)))
                    # add region
                    getattr(page, 'add_%s' % classname)(region)

        return OcrdPageResult(pcgts)
