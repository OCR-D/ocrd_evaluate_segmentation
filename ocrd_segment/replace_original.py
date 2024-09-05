from __future__ import absolute_import

from typing import Optional
from ocrd_utils import (
    coordinates_of_segment,
    points_from_polygon,
)
from ocrd_models.ocrd_page import (
    OcrdPage,
    AlternativeImageType,
    TextRegionType,
)
from ocrd import Processor, OcrdPageResult, OcrdPageResultImage

from .repair import ensure_valid

class ReplaceOriginal(Processor):

    @property
    def executable(self):
        return 'ocrd-segment-replace-original'

    def process_page_pcgts(self, *input_pcgts: Optional[OcrdPage], page_id: Optional[str] = None) -> OcrdPageResult:
        """Extract page image and replace original with it.

        Open and deserialize PAGE input file and its respective images,
        then go to the page hierarchy level.

        Retrieve the image of the (cropped, deskewed, dewarped) page, preferring
        the last annotated form (which, depending on the workflow, could be
        binarized or raw). Add that image file to the workspace with the fileGrp
        USE given in the output fileGrp.
        Reference that file in the page (not as AlternativeImage but) as original
        image. Adjust all segment coordinates accordingly.

        Produce a new output file by serialising the resulting hierarchy.
        """
        pcgts = input_pcgts[0]
        result = OcrdPageResult(pcgts)
        page = pcgts.get_Page()
        page_image, page_coords, page_image_info = self.workspace.image_from_page(
            page, page_id,
            feature_filter=self.parameter['feature_filter'],
            feature_selector=self.parameter['feature_selector'],
        )
        # annotate extracted image as "new original"
        result.images.append(OcrdPageResultImage(page_image, '.IMG', page))
        page.set_Border(None) # also removes all derived images
        page.set_orientation(None)
        # also add image as derived image (in order to preserve image features)
        # (but exclude coordinate-sensitive features that have already been applied over the "original")
        features = ','.join(filter(lambda f: f not in [
            "cropped", "deskewed", "rotated-90", "rotated-180", "rotated-270"],
                                   page_coords['features'].split(",")))
        alt_image = AlternativeImageType(comments=features)
        page.add_AlternativeImage(alt_image)
        result.images.append(OcrdPageResultImage(page_image, '.IMG-COPY', alt_image))
        # adjust all coordinates
        if self.parameter['transform_coordinates']:
            for region in page.get_AllRegions():
                region_polygon = coordinates_of_segment(region, page_image, page_coords)
                region.get_Coords().set_points(points_from_polygon(region_polygon))
                ensure_valid(region)
                if isinstance(region, TextRegionType):
                    for line in region.get_TextLine():
                        line_polygon = coordinates_of_segment(line, page_image, page_coords)
                        line.get_Coords().set_points(points_from_polygon(line_polygon))
                        ensure_valid(line)
                        for word in line.get_Word():
                            word_polygon = coordinates_of_segment(word, page_image, page_coords)
                            word.get_Coords().set_points(points_from_polygon(word_polygon))
                            ensure_valid(word)
                            for glyph in word.get_Glyph():
                                glyph_polygon = coordinates_of_segment(glyph, page_image, page_coords)
                                glyph.get_Coords().set_points(points_from_polygon(glyph_polygon))
                                ensure_valid(glyph)
        return result

