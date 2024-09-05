from __future__ import absolute_import

from typing import Optional

from ocrd_utils import (
    coordinates_for_segment,
    points_from_polygon,
    polygon_from_points,
)
from ocrd_models.ocrd_page import OcrdPage, TextRegionType
from ocrd import Processor, OcrdPageResult

from .repair import ensure_consistent

class ReplacePage(Processor):

    @property
    def executable(self):
        return 'ocrd-segment-replace-page'

    def process_page_pcgts(self, *input_pcgts: Optional[OcrdPage], page_id: Optional[str] = None) -> OcrdPageResult:
        """Replace everything below the page level with another annotation.

        Open and deserialize PAGE input file from both input file groups,
        then go to the page hierarchy level.

        Replace all regions (and their reading order) from the page of
        the first input file group with all regions from the page of
        the second input file group. Keep page-level annotations unchanged
        (i.e. Border, orientation, type, AlternativeImage etc).

        If ``transform_coordinates`` is true, then also retrieve the
        coordinate transform of the (cropped, deskewed, dewarped) page
        from the first input fileGrp, and use it to adjust all segment
        coordinates from the second input fileGrp, accordingly.
        (This assumes both are consistent, i.e. the second input was derived
        from the first input via ``ocrd-segment-replace-original`` or similar.)

        Produce a new output file by serialising the resulting hierarchy.
        """
        pcgts = input_pcgts[0]
        page = pcgts.get_Page()
        pcgts2 = input_pcgts[1]
        page2 = pcgts2.get_Page()
        if self.parameter['transform_coordinates']:
            # adjust all coordinates (recursively)
            try:
                _, page_coords, _ = self.workspace.image_from_page(page, page_id)
                for region in page2.get_AllRegions():
                    region_coords = region.get_Coords()
                    region_polygon = polygon_from_points(region_coords.points)
                    region_polygon = coordinates_for_segment(region_polygon, None, page_coords)
                    region_coords.set_points(points_from_polygon(region_polygon))
                    ensure_consistent(region)
                    if isinstance(region, TextRegionType):
                        for line in region.get_TextLine():
                            line_coords = line.get_Coords()
                            line_polygon = polygon_from_points(line_coords.points)
                            line_polygon = coordinates_for_segment(line_polygon, None, page_coords)
                            line_coords.set_points(points_from_polygon(line_polygon))
                            ensure_consistent(line)
                            for word in line.get_Word():
                                word_coords = word.get_Coords()
                                word_polygon = polygon_from_points(word_coords.points)
                                word_polygon = coordinates_for_segment(word_polygon, None, page_coords)
                                word_coords.set_points(points_from_polygon(word_polygon))
                                ensure_consistent(word)
                                for glyph in word.get_Glyph():
                                    glyph_coords = glyph.get_Coords()
                                    glyph_polygon = polygon_from_points(glyph_coords.points)
                                    glyph_polygon = coordinates_for_segment(glyph_polygon, None, page_coords)
                                    glyph_coords.set_points(points_from_polygon(glyph_polygon))
                                    ensure_consistent(glyph)
            except:
                self.logger.error('invalid coordinates on page %s', page_id)
                raise
        # replace all regions
        page.set_ReadingOrder(page2.get_ReadingOrder())
        page.set_TextRegion(page2.get_TextRegion())
        page.set_ImageRegion(page2.get_ImageRegion())
        page.set_LineDrawingRegion(page2.get_LineDrawingRegion())
        page.set_GraphicRegion(page2.get_GraphicRegion())
        page.set_TableRegion(page2.get_TableRegion())
        page.set_ChartRegion(page2.get_ChartRegion())
        page.set_MapRegion(page2.get_MapRegion())
        page.set_SeparatorRegion(page2.get_SeparatorRegion())
        page.set_MathsRegion(page2.get_MathsRegion())
        page.set_ChemRegion(page2.get_ChemRegion())
        page.set_MusicRegion(page2.get_MusicRegion())
        page.set_AdvertRegion(page2.get_AdvertRegion())
        page.set_NoiseRegion(page2.get_NoiseRegion())
        page.set_UnknownRegion(page2.get_UnknownRegion())
        page.set_CustomRegion(page2.get_CustomRegion())
        return OcrdPageResult(pcgts)
