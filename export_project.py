# Migrate a project

import os
import sys
import time
import pathlib

import argparse
import json
import requests

import xml.etree.ElementTree as ET
from PIL import Image
from io import BytesIO
from xml.dom import minidom

from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.training.models import ImageUrlCreateBatch, ImageUrlCreateEntry, Region


def migrate_tags(src_trainer, dest_trainer, project_id, dest_project):
    tags =  src_trainer.get_tags(project_id)
    print ("Found:", len(tags), "tags")
    # Re-create all of the tags and store them for look-up
    created_tags = {}
    if dest_trainer:
        for tag in src_trainer.get_tags(project_id):
            print ("Creating tag:", tag.name, tag.id)
            created_tags[tag.id] = dest_trainer.create_tag(dest_project.id, tag.name, description=tag.description, type=tag.type).id
            return created_tags
    else:
        with open('export/tags.json', 'w') as fp:
            t = {x.id:x.name for x in tags}
            json.dump(t, fp)
        return t

def migrate_images(src_trainer, dest_trainer, project_id, dest_project, created_tags):
    # Migrate any tagged images that may exist and preserve their tags and regions.
    count = src_trainer.get_tagged_image_count(project_id)
    print ("Found:",count,"tagged images.")
    migrated = 0
    while(count > 0):
        count_to_migrate = min(count, 50)
        print ("Getting", count_to_migrate, "images")
        images = src_trainer.get_tagged_images(project_id, take=count_to_migrate, skip=migrated)
        images_to_upload = []
        for i in images:
            print ("Migrating", i.id, i.original_image_uri)
            if i.regions:
                regions = []
                tag_ids = []
                for r in i.regions:
                    print ("Found region:", r.region_id, r.tag_id, r.left, r.top, r.width, r.height)
                    regions.append(Region(tag_id=created_tags[r.tag_id], left=r.left, top=r.top, width=r.width, height=r.height))
                    tag_ids.append(created_tags[r.tag_id])
                entry = ImageUrlCreateEntry(url=i.original_image_uri, regions=regions)
            else:
                tag_ids = []
                for t in i.tags:
                    print ("Found tag:", t.tag_name, t.tag_id)
                    tag_ids.append(created_tags[t.tag_id])
                entry = ImageUrlCreateEntry(url=i.original_image_uri, tag_ids=tag_ids)

            images_to_upload.append(entry)
            image_file = '{}-{}.jpg'.format(i.id, "_".join(tag_ids))
            xml_file = '{}-{}.xml'.format(i.id, "_".join(tag_ids))
            image = None
            if not os.path.exists('export/' + image_file):
                r = requests.get(i.original_image_uri)
                with open('export/' + image_file, 'wb') as f:
                    f.write(r.content)
                    image = Image.open(BytesIO(r.content))
            else:
                image = Image.open('export/' + image_file)
            w, h = image.size
            annotation = ET.Element('annotation')
            folder = ET.SubElement(annotation, 'folder')
            filename = ET.SubElement(annotation, 'filename')
            filename.text = image_file
            path = ET.SubElement(annotation, 'path')
            path.text = image_file
            source = ET.SubElement(annotation, 'source')
            database = ET.SubElement(source, 'database')
            database.text = 'Egge'
            size = ET.SubElement(annotation, 'size')
            ET.SubElement(size, 'width').text = str(w)
            ET.SubElement(size, 'height').text = str(h)
            ET.SubElement(size, 'depth').text = '3'
            ET.SubElement(annotation, 'segmented').text = '0'
            for r in i.regions:
                _object = ET.SubElement(annotation, 'object')
                ET.SubElement(_object, 'name').text = created_tags[r.tag_id]
                ET.SubElement(_object, 'pose').text= 'Unspecified'
                ET.SubElement(_object, 'truncated').text = '0'
                ET.SubElement(_object, 'difficult').text = '0'
                ET.SubElement(_object, 'occluded').text = '0'
                bndbox = ET.SubElement(_object, 'bndbox')
                ET.SubElement(bndbox, 'xmin').text = str(int(r.left * w))
                ET.SubElement(bndbox, 'xmax').text = str(int((r.left + r.width) * w))
                ET.SubElement(bndbox, 'ymin').text = str(int(r.top * h))
                ET.SubElement(bndbox, 'ymax').text = str(int((r.top + r.height) * h))
            xmlstr = minidom.parseString(ET.tostring(annotation)).toprettyxml(indent="   ")
            with open('export/' + xml_file, "wb") as f:
                f.write(xmlstr.encode('utf-8'))

        if dest_trainer:
            upload_result = dest_trainer.create_images_from_urls(dest_project.id, images=images_to_upload)
            if not upload_result.is_batch_successful:
                print ("ERROR: Failed to upload image batch")
                for i in upload_result.images:
                    print ("\tImage status:", i.id, i.status)
                exit(-1)

        migrated += count_to_migrate
        count -= count_to_migrate

    # Migrate any untagged images that may exist.
    count = src_trainer.get_untagged_image_count(project_id)
    print ("Found:", count, "untagged images.")
    migrated = 0
    while(count > 0):
        count_to_migrate = min(count, 50)
        print ("Getting", count_to_migrate, "images")
        images = src_trainer.get_untagged_images(project_id, take=count_to_migrate, skip=migrated)
        images_to_upload = []
        for i in images:
            print ("Migrating", i.id, i.original_image_uri)
            images_to_upload.append(ImageUrlCreateEntry(url=i.original_image_uri))

        upload_result = dest_trainer.create_images_from_urls(dest_project.id, images=images_to_upload)
        if not upload_result.is_batch_successful:
            print ("ERROR: Failed to upload image batch")
            for i in upload_result.images:
                print ("\tImage status:", i.id, i.status)
            exit(-1)
        migrated += count_to_migrate
        count -= count_to_migrate
    return images

def migrate_project(src_trainer, dest_trainer, project_id):
    # Get the original project
    src_project = src_trainer.get_project(project_id)
    print ("Source project:", src_project.name)
    print ("\tDescription:", src_project.description)
    print ("\tDomain:", src_project.settings.domain_id)
    if src_project.settings.classification_type:
        print ("\tClassificationType:", src_project.settings.classification_type)
    print("\tTarget Export Platforms:", src_project.settings.target_export_platforms)

    # Create the destination project
    return dest_trainer.create_project(src_project.name, description=src_project.description, domain_id=src_project.settings.domain_id, classification_type=src_project.settings.classification_type, target_export_platforms=src_project.settings.target_export_platforms)

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-p", "--project", action="store", type=str, help="Source project ID", dest="project_id", default=None)
    arg_parser.add_argument("-s", "--src", action="store", type=str, help="Source Training-Key", dest="source_training_key", default=None)
    arg_parser.add_argument("-se", "--src_endpoint", action="store", type=str, help="Source Endpoint", dest="source_endpoint", default="https://southcentralus.api.cognitive.microsoft.com")
    arg_parser.add_argument("-x", "--export", action='store_true', help="Export project locally in Pascal VOC format")
    arg_parser.add_argument("-d", "--dest", action="store", type=str, help="Destination Training-Key", dest="destination_training_key", default=None)
    arg_parser.add_argument("-de", "--dest_endpoint", action="store", type=str, help="Destination Endpoint", dest="destination_endpoint", default="https://southcentralus.api.cognitive.microsoft.com")
    args = arg_parser.parse_args()

    if (not args.project_id or not args.source_training_key):
        arg_parser.print_help()
        exit(-1)

    print ("Collecting information for source project:", args.project_id)

    # Client for Source
    src_trainer = CustomVisionTrainingClient(args.source_training_key, endpoint=args.source_endpoint)

    # Client for Destination
    if args.export:
        pathlib.Path("export").mkdir(parents=True, exist_ok=True)
        dest_trainer = None
    else:
        dest_trainer = CustomVisionTrainingClient(args.destination_training_key, endpoint=args.destination_endpoint)

    if dest_trainer:
        destination_project = migrate_project(src_trainer, dest_trainer, args.project_id)
    else:
        destination_project = None
    tags = migrate_tags(src_trainer, dest_trainer, args.project_id, destination_project)
    source_images = migrate_images(src_trainer, dest_trainer, args.project_id, destination_project, tags)
