from __future__ import absolute_import

from typing import Optional
import json
import itertools

from ocrd_utils import (
    config,
    make_file_id,
    coordinates_of_segment,
    polygon_from_points,
    MIME_TO_EXT
)
from ocrd_modelfactory import page_from_file
from ocrd_models.ocrd_file import OcrdFileType
from ocrd import Processor


class ExtractWords(Processor):

    @property
    def executable(self):
        return 'ocrd-segment-extract-words'

    def process_page_file(self, *input_files : Optional[OcrdFileType]) -> None:
        """Extract word images and texts from the workspace.

        Open and deserialize PAGE input files and their respective images,
        then iterate over the element hierarchy down to the word level.

        Extract an image for each word (which depending on the workflow
        can already be deskewed, dewarped, binarized etc.), cropped to its
        minimal bounding box, and masked by the coordinate polygon outline.
        Apply ``feature_filter`` (a comma-separated list of image features,
        cf. :py:func:`ocrd.workspace.Workspace.image_from_page`) to skip
        specific features when retrieving derived images.
        If ``transparency`` is true, then also add an alpha channel which is
        fully transparent outside of the mask.

        \b
        Create a JSON file with:
        * the IDs of the word and its parents,
        * the word's text content,
        * the word's coordinates relative to the line image,
        * the word's absolute coordinates,
        * the word's TextStyle (if any),
        * the word's @production (if any),
        * the word's @readingDirection (if any),
        * the word's @primaryScript (if any),
        * the word's @language (if any),
        * the word's AlternativeImage/@comments (features),
        * the parent textregion's @type,
        * the page's @type,
        * the page's DPI value.

        Create a plain text file for the text content, too.

        \b
        Write all files in the directory of the output file group, named like so:
        * ID + '.raw.png': word image (if the workflow provides raw images)
        * ID + '.bin.png': word image (if the workflow provides binarized images)
        * ID + '.nrm.png': word image (if the workflow provides grayscale-normalized images)
        * ID + '.json': word metadata.
        * ID + '.gt.txt': word text.

        (This is intended for training and evaluation of OCR models.)
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
                    lpolygon_rel = coordinates_of_segment(
                        word, word_image, word_coords).tolist()
                    lpolygon_abs = polygon_from_points(word.get_Coords().points)
                    ltext = word.get_TextEquiv()
                    if not ltext:
                        self.logger.warning("Word '%s' contains no text content", word.id)
                        ltext = ''
                    else:
                        ltext = ltext[0].Unicode
                    lstyle = word.get_TextStyle() or line.get_TextStyle() or region.get_TextStyle()
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
                    lfeatures = word_coords['features']
                    description = { 'word.ID': word.id,
                                    'text': ltext,
                                    'style': lstyle,
                                    'production': (
                                        word.get_production() or
                                        line.get_production() or
                                        region.get_production()),
                                    'readingDirection': (
                                        word.get_readingDirection() or
                                        line.get_readingDirection() or
                                        region.get_readingDirection() or
                                        page.get_readingDirection()),
                                    'primaryScript': (
                                        word.get_primaryScript() or
                                        line.get_primaryScript() or
                                        region.get_primaryScript() or
                                        page.get_primaryScript()),
                                    'language': (
                                        word.get_language() or
                                        line.get_primaryLanguage() or
                                        region.get_primaryLanguage() or
                                        page.get_primaryLanguage()),
                                    'features': lfeatures,
                                    'DPI': dpi,
                                    'coords_rel': lpolygon_rel,
                                    'coords_abs': lpolygon_abs,
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
                        word_image,
                        file_id + '_' + region.id + '_' + line.id + '_' + word.id + extension,
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
