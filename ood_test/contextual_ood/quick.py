import gspread
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from oauth2client.service_account import ServiceAccountCredentials

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

creds = ServiceAccountCredentials.from_json_keyfile_name('/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/credential.json', scope)

client = gspread.authorize(creds)

sheet = client.open_by_key('1Xt4P21X9I9tH_u5H-EHTzeMeM5X-oRCyAeQDtTSg-oU').sheet1  # sheet1 refers to the first sheet

SERVICE_ACCOUNT_FILE = '/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/credential.json'
SCOPES = ['https://www.googleapis.com/auth/drive.file']
credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES)
service = build('drive', 'v3', credentials=credentials)
file_path = '/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/coco/185250.jpg'
spreadsheet = client.open_by_key('1Xt4P21X9I9tH_u5H-EHTzeMeM5X-oRCyAeQDtTSg-oU')
sheet = spreadsheet.worksheet(f"joint_lastl_top")
list_of_lists = sheet.get_all_values()
last_row = len(list_of_lists)
print("sheet ", last_row)
sheet.update_cell(last_row + 1, 1, "helloworld")

folder_metadata = {
    'name': 'joint_last_l',
    'mimeType': 'application/vnd.google-apps.folder'
}
folder = service.files().create(
    body=folder_metadata,
    fields='id'
).execute()
folder_id = folder.get('id')
print(f'Folder ID: {folder_id}')

# Upload an image to the folder
file_metadata = {
    'name': 'image.jpg',
    'parents': [folder_id]  # Specify the folder ID
}
media = MediaFileUpload(file_path, mimetype='image/jpeg')
file = service.files().create(
    body=file_metadata,
    media_body=media,
    fields='id'
).execute()
file_id = file.get('id')
print(f'File ID: {file_id}')

# Set folder permissions to make it accessible to anyone with the link
permission = {
    'type': 'anyone',
    'role': 'reader'
}
service.permissions().create(
    fileId=folder_id,
    body=permission
).execute()
print('Folder permissions set to public.')

# Create a shareable link
folder_link = f'https://drive.google.com/drive/folders/{folder_id}'
print(f'Folder URL: {folder_link}')
file_link = f'https://drive.google.com/uc?id={file_id}'
print("file link", file_link)