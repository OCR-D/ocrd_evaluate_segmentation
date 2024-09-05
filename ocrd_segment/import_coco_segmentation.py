from __future__ import absolute_import

from typing import Optional
import os.path
import json
import logging

import numpy as np

from ocrd_utils import (
    points_from_polygon,
    MIMETYPE_PAGE,
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
    to_xml)
from ocrd_models.ocrd_page_generateds import (
    BorderType,
    TableRegionType,
    GraphicRegionType,
    ChartRegionType,
    ChemRegionType,
    LineDrawingRegionType,
    MusicRegionType,
    UnknownRegionType,
    AdvertRegionType,
    CustomRegionType,
    MapRegionType,
    TextTypeSimpleType,
    GraphicsTypeSimpleType,
    ChartTypeSimpleType
)
# pragma pylint: enable=unused-import
from ocrd import Workspace, Processor, OcrdPageResult

TYPEDICT = {
    "TextRegion": TextTypeSimpleType,
    "GraphicRegion": GraphicsTypeSimpleType,
    "ChartType": ChartTypeSimpleType
}


class ImportCOCOSegmentation(Processor):

    @property
    def executable(self):
        return 'ocrd-segment-from-coco'

    def process_workspace(self, workspace: Workspace) -> None:
        """Performs region segmentation by reading from COCO annotations.

        Open and deserialize the COCO JSON file from the second input file group.
        (It lists region categories/subtypes, file names and segmentations for all pages.)

        \b
        Open and deserialize each PAGE input file (or generate from image input file)
        from the first input file group. Now find this page in COCO:
        - try to match the PAGE ``imageFilename`` or METS file path matches to some
          COCO ``file_name``, otherwise
        - try to match the numeric part of the METS physical page ID to some
          COCO ``id``, otherwise
        - skip with an error.

        Then create and add a region for each ``segmentation``, converting its polygon
        to coordinate points and its COCO category to a region type (and subtype),
        either for a PubLayNet classification or PAGE classification (as produced by
        ocrd-segment-extract-pages), as indicated by ``source``.

        Produce a new output file by serialising the resulting hierarchy.

        Afterwards, if there are still COCO images left unaccounted for (i.e. without
        corresponding input files), then show a warning.
        """
        # overwrite input_file_grp to single to prevent zip_input_files from searching page pairs
        #self.verify()
        self.input_file_grp, coco_grp = self.input_file_grp.split(',')
        # make sure the cardinality requirement is also reduced from 2 to 1
        self.ocrd_tool['input_file_grp_cardinality'] = 1
        # Load JSON
        if coco_grp in workspace.mets.file_groups:
            try:
                cocofile = next(f for f in workspace.mets.find_files(fileGrp=coco_grp)
                                # if f.mimetype == 'application/json' and not f.pageId
                                if not f.pageId)
                if self.download:
                    cocofile = workspace.download_file(cocofile)
                cocofile = os.path.join(workspace.directory, cocofile.local_filename)
            except StopIteration:
                raise Exception("no document-wide file (COCO JSON file) in second file group", coco_grp)
        elif os.path.isfile(os.path.join(workspace.directory, coco_grp)):
            # passing a path as input fileGrp is not strictly allowed in OCR-D
            cocofile = os.path.join(workspace.directory, coco_grp)
        else:
            raise Exception("file not found in second file group (COCO file)", coco_grp)

        self.logger.info('Loading COCO annotations from "%s" into memory...', cocofile)
        with open(cocofile, 'r') as cocof:
            coco = json.load(cocof)
        self.logger.info('Loaded JSON for %d images with %d regions in %d categories',
                 len(coco['images']), len(coco['annotations']), len(coco['categories']))
        # Convert to usable dicts
        # classes:
        self.coco_source = 'custom'
        self.categories = {}
        self.subcategories = {}
        for cat in coco['categories']:
            if 'source' in cat:
                self.coco_source = cat['source']
            if 'supercategory' in cat and cat['supercategory']:
                self.categories[cat['id']] = cat['supercategory']
                self.subcategories[cat['id']] = cat['name']
            else:
                self.categories[cat['id']] = cat['name']
        # images and annotations:
        self.images_by_id = {}
        self.images_by_filename = {}
        for image in coco['images']:
            self.images_by_id[image['id']] = image
            self.images_by_filename[image['file_name']] = image
        for annotation in coco['annotations']:
            image = self.images_by_id[annotation['image_id']]
            image.setdefault('regions', []).append(annotation)
        del coco
        self.logger.info('Converting %s annotations into PAGE-XML', self.coco_source)
        super().process_workspace(workspace)

        # warn of remaining COCO images
        if self.images_by_filename and not self.page_id:
            self.logger.warning('%d images remain unaccounted for after processing', len(self.images_by_filename))
            if self.logger.isEnabledFor(logging.DEBUG):
                for filename in self.images_by_filename:
                    self.logger.debug('not found in workspace: "%s"', filename)

    def process_page_pcgts(self, *input_pcgts: Optional[OcrdPage], page_id: Optional[str] = None) -> OcrdPageResult:
        try:
            # separate non-numeric part of page ID to retain the numeric part
            num_page_id = int(page_id.strip(page_id.strip("0123456789")))
        except Exception:
            num_page_id = self.workspace.mets.physical_pages.index(page_id)
        pcgts = input_pcgts[0]
        page = pcgts.get_Page()

        # find COCO image
        if page.imageFilename in self.images_by_filename:
            image = self.images_by_filename[page.imageFilename]
        elif os.path.basename(page.imageFilename) in self.images_by_filename:
            image = self.images_by_filename[os.path.basename(page.imageFilename)]
        elif num_page_id in self.images_by_id:
            image = self.images_by_id[num_page_id]
        else:
            raise Exception(f'Page "{page_id}" / file "{page.imageFilename}" not found in COCO')

        if image['width'] != page.imageWidth:
            self.logger.error('Page "%s" width %d does not match annotated width %d',
                              page_id, page.imageWidth, image['width'])
        if image['height'] != page.imageHeight:
            self.logger.error('Page "%s" height %d does not match annotated height %d',
                              page_id, page.imageHeight, image['height'])

        # todo: remove existing segmentation first?
        for region in image.get('regions', []):
            assert isinstance(region['segmentation'], list), "importing RLE/mask segmentation not implemented"
            polygon = np.array(region['segmentation'])
            polygon = np.reshape(polygon, (polygon.shape[1]//2, 2))
            coords = CoordsType(points=points_from_polygon(polygon))
            category = self.categories[region['category_id']]
            if region['category_id'] in self.subcategories:
                subcategory = self.subcategories[region['category_id']]
            else:
                subcategory = None
            if subcategory == category:
                subcategory = None
            mapping = self.parameter['categorydict']
            region_id = f"r{region['id']}"
            self.logger.info('Adding region %s:%s [area %d]', category, subcategory or '', region['area'])
            args = {'id': region_id,
                    'Coords': coords}
            if self.coco_source != 'PAGE':
                if subcategory:
                    category = mapping[category + ':' + subcategory]
                else:
                    category = mapping[category]
                if ':' in category:
                    category, subcategory = category.split(':')
                else:
                    subcategory = None
            if subcategory:
                if category in TYPEDICT:
                    subtype = membername(TYPEDICT[category], subcategory)
                    if subtype == subcategory:
                        # not predefined in PAGE: use other + custom
                        args['custom'] = "subtype:%s" % subcategory
                        args['type_'] = "other"
                    else:
                        args['type_'] = subcategory
                else:
                    args['custom'] = "subtype:%s" % subcategory
            if category + 'Type' not in globals():
                raise Exception('unknown region category: %s' % category)
            region_type = globals()[category + 'Type']
            if region_type is BorderType:
                page.set_Border(BorderType(Coords=coords))
            else:
                region_obj = region_type(**args)
                getattr(page, 'add_%s' % category)(region_obj)

        # remove image from dicts (so lookup becomes faster and we know if anything remains unaccounted for)
        self.images_by_id.pop(num_page_id, None)
        self.images_by_filename.pop(page.imageFilename, None)

        return OcrdPageResult(pcgts)
