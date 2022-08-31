from __future__ import print_function

import google.auth
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import json
from googleapiclient.http import MediaFileUpload
from oauth2client.service_account import ServiceAccountCredentials


def create_image(presentation_id, page_id, creds, url):
    """
        Creates images the user has access to.
        Load pre-authorized user credentials from the environment.
        TODO(developer) - See https://developers.google.com/identity
        for guides on implementing OAuth2 for the application.
        """

    # pylint: disable=maybe-no-member
    try:
        service = build('slides', 'v1', credentials=creds)
        # pylint: disable = invalid-name
        IMAGE_URL = (url)
        # pylint: disable=invalid-name
        requests = []
        image_id = 'MyImage_12'
        emu4M = {
            'magnitude': 3800000,
            'unit': 'EMU'
        }
        requests.append({"deleteObject": {
            "objectId": image_id}
        })
        try:
            body = {
                'requests': requests
            }
            response = service.presentations() \
                .batchUpdate(presentationId=presentation_id, body=body).execute()
        except HttpError as error:
            print(f"Error deleting: {error}")

        requests = []
        requests.append(
            {

                'createImage': {
                    'objectId': image_id,
                    'url': IMAGE_URL,
                    'elementProperties': {
                        'pageObjectId': page_id,
                        'size': {
                            'height': emu4M,
                            'width': emu4M
                        },
                        'transform': {
                            'scaleX': 1,
                            'scaleY': 1,
                            'translateX': 0,
                            'translateY': 1100000,
                            'unit': 'EMU'
                        }
                    }
                }
            })

        # Execute the request.
        body = {
            'requests': requests
        }
        response = service.presentations() \
            .batchUpdate(presentationId=presentation_id, body=body).execute()
        create_image_response = response.get('replies')[0].get('createImage')
        print(f"Created image with ID: "
              f"{(create_image_response.get('objectId'))}")

        return response
    except HttpError as error:
        print(f"An error occurred: {error}")
        print("Images not created")
        return error


def upload_basic(creds, fileName):
    """Insert new file.
    Returns : Id's of the file uploaded

    Load pre-authorized user credentials from the environment.
    TODO(developer) - See https://developers.google.com/identity
    for guides on implementing OAuth2 for the application.
    """

    try:
        # create drive api client
        service = build('drive', 'v3', credentials=creds)

        file_metadata = {'name': fileName}
        media = MediaFileUpload(fileName,
                                mimetype='image/png')
        # pylint: disable=maybe-no-member
        file = service.files().create(body=file_metadata, media_body=media,
                                      fields='id').execute()
        print(F'File ID: {file.get("id")}')

        service.permissions().create(fileId=file.get("id"), body={
            "role": 'reader',
            "type": 'anyone',
        }).execute()

        # link = service.files().get(
        #    fileId=file.get("id"),
        #    fields='webViewLink'
        # ).execute()
        # print(link)
        return file.get("id")

    except HttpError as error:
        print(F'An error occurred: {error}')
        file = None

    return file.get('id')


def write(key_file_dict, fileName):
    scope = ['https://spreadsheets.google.com/feeds',
             'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_dict(
        key_file_dict, scope)

    id = upload_basic(creds, fileName)
    link = "https://drive.google.com/uc?export=view&id=" + id
    create_image("1T98K0Ci13GkN6R55Op3JTbcaxiNtgi_4oNy6NdV-hHs",
                 "g149273082de_1_288", creds, link)
