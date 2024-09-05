from __future__ import absolute_import

from typing import Optional
import json
import itertools

from ocrd_utils import (
    config,
    make_file_id,
    assert_file_grp_cardinality,
    coordinates_of_segment,
    polygon_from_points,
    MIME_TO_EXT
)
from ocrd_modelfactory import page_from_file
from ocrd_models.ocrd_file import OcrdFileType
from ocrd import Processor


class ExtractGlyphs(Processor):

    @property
    def executable(self):
        return 'ocrd-segment-extract-glyphs'

    def process_page_file(self, *input_files : Optional[OcrdFileType]) -> None:
        """Extract glyph images and texts from the workspace.

        Open and deserialize PAGE input files and their respective images,
        then iterate over the element hierarchy down to the glyph level.

        Extract an image for each glyph (which depending on the workflow
        can already be deskewed, dewarped, binarized etc.), cropped to its
        minimal bounding box, and masked by the coordinate polygon outline.
        Apply ``feature_filter`` (a comma-separated list of image features,
        cf. :py:func:`ocrd.workspace.Workspace.image_from_page`) to skip
        specific features when retrieving derived images.
        If ``transparency`` is true, then also add an alpha channel which is
        fully transparent outside of the mask.

        \b
        Create a JSON file with:
        * the IDs of the glyph and its parents,
        * the glyph's text content,
        * the glyph's coordinates relative to the line image,
        * the glyph's absolute coordinates,
        * the glyph's TextStyle (if any),
        * the glyph's @production (if any),
        * the glyph's @ligature (if any),
        * the glyph's @symbol (if any),
        * the glyph's @script (if any),
        * the glyph's AlternativeImage/@comments (features),
        * the parent textregion's @type,
        * the page's @type,
        * the page's DPI value.

        Create a plain text file for the text content, too.

        \b
        Write all files in the directory of the output file group, named like so:
        * ID + '.raw.png': glyph image (if the workflow provides raw images)
        * ID + '.bin.png': glyph image (if the workflow provides binarized images)
        * ID + '.nrm.png': glyph image (if the workflow provides grayscale-normalized images)
        * ID + '.json': glyph metadata.
        * ID + '.gt.txt': glyph text.

        (This is intended for training and evaluation of script detection models.)
        """
        input_file = input_files[0]
        page_id = input_file.pageId
        try:
            pcgts = page_from_file(input_file)
        except ValueError as err:
            # not PAGE and not an image to generate PAGE for
            self.logger.error(f"non-PAGE input for page {page_id}: {err}")
            raise

        page = pcgts.get_Page()
        page_image, page_coords, page_image_info = self.workspace.image_from_page(
            page, page_id,
            feature_filter=self.parameter['feature_filter'],
            transparency=self.parameter['transparency'])
        if page_image_info.resolution != 1:
            dpi = page_image_info.resolution
            if page_image_info.resolutionUnit == 'cm':
                dpi = round(dpi * 2.54)
        else:
            dpi = None
        ptype = page.get_type()

        regions = itertools.chain.from_iterable(
            [page.get_TextRegion()] +
            [subregion.get_TextRegion() for subregion in page.get_TableRegion()])
        if not regions:
            self.logger.warning("Page '%s' contains no text regions", page_id)
        for region in regions:
            region_image, region_coords = self.workspace.image_from_segment(
                region, page_image, page_coords,
                feature_filter=self.parameter['feature_filter'],
                transparency=self.parameter['transparency'])
            rtype = region.get_type()

            lines = region.get_TextLine()
            if not lines:
                self.logger.warning("Region '%s' contains no text lines", region.id)
            for line in lines:
                line_image, line_coords = self.workspace.image_from_segment(
                    line, region_image, region_coords,
                    feature_filter=self.parameter['feature_filter'],
                    transparency=self.parameter['transparency'])
                words = line.get_Word()
                if not words:
                    self.logger.warning("Line '%s' contains no words", line.id)
                for word in words:
                    word_image, word_coords = self.workspace.image_from_segment(
                        word, line_image, line_coords,
                        feature_filter=self.parameter['feature_filter'],
                        transparency=self.parameter['transparency'])
                    glyphs = word.get_Glyph()
                    if not glyphs:
                        self.logger.warning("Word '%s' contains no glyphs", word.id)
                    for glyph in glyphs:
                        glyph_image, glyph_coords = self.workspace.image_from_segment(
                            glyph, word_image, word_coords,
                            feature_filter=self.parameter['feature_filter'],
                            transparency=self.parameter['transparency'])
                        lpolygon_rel = coordinates_of_segment(
                            glyph, glyph_image, glyph_coords).tolist()
                        lpolygon_abs = polygon_from_points(glyph.get_Coords().points)
                        ltext = glyph.get_TextEquiv()
                        if not ltext:
                            self.logger.warning("Glyph '%s' contains no text content", glyph.id)
                            ltext = ''
                        else:
                            ltext = ltext[0].Unicode
                        lstyle = glyph.get_TextStyle() or word.get_TextStyle() or line.get_TextStyle() or region.get_TextStyle()
                        if lstyle:
                            lstyle = {
                                'fontFamily': lstyle.fontFamily,
                                'fontSize': lstyle.fontSize,
                                'xHeight': lstyle.xHeight,
                                'kerning': lstyle.kerning,
                                'serif': lstyle.serif,
                                'monospace': lstyle.monospace,
                                'bold': lstyle.bold,
                                'italic': lstyle.italic,
                                'smallCaps': lstyle.smallCaps,
                                'letterSpaced': lstyle.letterSpaced,
                                'strikethrough': lstyle.strikethrough,
                                'underlined': lstyle.underlined,
                                'underlineStyle': lstyle.underlineStyle,
                                'subscript': lstyle.subscript,
                                'superscript': lstyle.superscript
                            }
                        lfeatures = glyph_coords['features']
                        description = { 'glyph.ID': glyph.id,
                                        'text': ltext,
                                        'style': lstyle,
                                        'production': (
                                            glyph.get_production() or
                                            word.get_production() or
                                            line.get_production() or
                                            region.get_production()),
                                        'script': (
                                            glyph.get_script() or
                                            word.get_primaryScript() or
                                            line.get_primaryScript() or
                                            region.get_primaryScript() or
                                            page.get_primaryScript()),
                                        'ligature': glyph.get_ligature(),
                                        'symbol': glyph.get_symbol(),
                                        'features': lfeatures,
                                        'DPI': dpi,
                                        'coords_rel': lpolygon_rel,
                                        'coords_abs': lpolygon_abs,
                                        'word.ID': word.id,
                                        'line.ID': line.id,
                                        'region.ID': region.id,
                                        'region.type': rtype,
                                        'page.ID': page_id,
                                        'page.type': ptype,
                                        'file_grp': self.input_file_grp,
                                        'METS.UID': self.workspace.mets.unique_identifier
                        }
                        if 'binarized' in lfeatures:
                            extension = '.bin'
                        elif 'grayscale_normalized' in lfeatures:
                            extension = '.nrm'
                        else:
                            extension = '.raw'

                        file_id = make_file_id(input_file, self.output_file_grp)
                        file_path = self.workspace.save_image_file(
                            glyph_image,
                            file_id + '_' + region.id + '_' + line.id + '_' + word.id + '_' + glyph.id + extension,
                            self.output_file_grp,
                            page_id=page_id,
                            mimetype=self.parameter['mimetype'],
                            force=config.OCRD_EXISTING_OUTPUT == 'OVERWRITE',
                        )
                        file_path = file_path.replace(extension + MIME_TO_EXT[self.parameter['mimetype']], '.json')
                        json.dump(description, open(file_path, 'w'), indent=2)
                        file_path = file_path.replace('.json', '.gt.txt')
                        with open(file_path, 'wb') as f:
                            f.write((ltext + '\n').encode('utf-8'))
