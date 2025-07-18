import hashlib
import hmac
import json
import requests
import base64
import time
import random
import paramiko
import urllib.parse
import pytz
import datetime
from requests_oauthlib import OAuth1
from urllib.parse import urlparse
import xml.etree.ElementTree as ET
import re
import io
from collections import defaultdict
import os
import logging

logger = logging.getLogger(__name__)

_cached_secrets = None
_cached_queries = None

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_local_path(filename):
    return os.path.join(CURRENT_DIR, filename)


def get_secrets():
    global _cached_secrets
    if _cached_secrets is None:
        try:
            secrets_path = get_local_path("secrets.json")
            if os.path.exists(secrets_path):
                with open(secrets_path, "r") as f:
                    _cached_secrets = json.load(f)
                    logger.info("Loaded secrets from file")
            else:
                logger.warning(f"Secrets file not found at {secrets_path}, using empty dictionary")
                _cached_secrets = {}
        except Exception as e:
            logger.error(f"Error loading secrets: {str(e)}")
            _cached_secrets = {}
    return _cached_secrets


def get_queries():
    global _cached_queries
    if _cached_queries is None:
        try:
            queries_path = get_local_path("queries.json")
            if os.path.exists(queries_path):
                with open(queries_path, "r") as f:
                    _cached_queries = json.load(f)
                    logger.info("Loaded queries from file")
            else:
                logger.warning(f"Queries file not found at {queries_path}, using empty dictionary")
                _cached_queries = {}
        except Exception as e:
            logger.error(f"Error loading queries: {str(e)}")
            _cached_queries = {}
    return _cached_queries


def generate_timestamp():
    """Generate the current timestamp."""
    return str(int(time.time()))


def generate_nonce(length=11):
    """Generate a pseudorandom nonce."""
    return ''.join([str(random.randint(0, 9)) for _ in range(length)])


def generate_signature(method, url, params, consumer_key, nonce, timestamp, token, consumer_secret, token_secret):
    # OAuth-specific params (added automatically)
    oauth_params = {
        "oauth_consumer_key": consumer_key,
        "oauth_token": token,
        "oauth_nonce": nonce,
        "oauth_timestamp": timestamp,
        "oauth_signature_method": "HMAC-SHA256",
        "oauth_version": "1.0"
    }

    # Merge user-provided query params and OAuth params
    all_params = {**params, **oauth_params}

    # Sort & percent-encode params
    sorted_items = sorted(
        (urllib.parse.quote_plus(str(k)), urllib.parse.quote_plus(str(v))) for k, v in all_params.items())
    param_string = '&'.join(f"{k}={v}" for k, v in sorted_items)

    # Construct the OAuth base string
    base_string = '&'.join([
        method.upper(),
        urllib.parse.quote(url, safe=''),
        urllib.parse.quote(param_string, safe='')
    ])

    # Construct the signing key
    signing_key = f"{urllib.parse.quote(consumer_secret)}&{urllib.parse.quote(token_secret)}"

    # HMAC-SHA256 signing
    hashed = hmac.new(signing_key.encode('utf-8'), base_string.encode('utf-8'), hashlib.sha256)
    signature = base64.b64encode(hashed.digest()).decode('utf-8')

    return signature


def query_netsuite(sql_query, limit=1000, offset=0):
    """
    Executes a SuiteQL query against the NetSuite REST API with OAuth 1.0a authentication.

    :param sql_query: The SuiteQL query to execute.
    :param limit: Number of records to return.
    :param offset: Offset for pagination.
    :return: JSON response from NetSuite.
    """
    # Load credentials from secrets file
    secrets = get_secrets()

    ns_account_id = secrets["realm"]
    consumer_key = secrets["ConsumerKey"]
    consumer_secret = secrets["ConsumerSecret"]
    token = secrets["ProdTokenID"]
    token_secret = secrets["ProdTokenSecret"]

    # Query parameters
    base_url = secrets["suiteQLurl"]
    params = {
        "limit": limit,
        "offset": offset
    }
    url_with_params = f"{base_url}?{urllib.parse.urlencode(params)}"

    # OAuth signature components
    nonce = generate_nonce()
    timestamp = generate_timestamp()

    signature = generate_signature(
        method="POST",
        url=base_url,  # Base URL only, no params
        params=params,  # Query params to be included in signature
        consumer_key=consumer_key,
        nonce=nonce,
        timestamp=timestamp,
        token=token,
        consumer_secret=consumer_secret,
        token_secret=token_secret
    )

    # Construct OAuth header
    oauth_header = (
        f'OAuth realm="{ns_account_id}", '
        f'oauth_consumer_key="{consumer_key}", '
        f'oauth_token="{token}", '
        f'oauth_signature_method="HMAC-SHA256", '
        f'oauth_timestamp="{timestamp}", '
        f'oauth_nonce="{nonce}", '
        f'oauth_version="1.0", '
        f'oauth_signature="{urllib.parse.quote(signature)}"'
    )

    # Set headers
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Prefer": "transient",
        "Authorization": oauth_header
    }

    # Query payload
    payload = json.dumps({
        "q": sql_query
    })
    print(f"SQL Query: {sql_query}")

    # Send the POST request to SuiteQL endpoint
    response = requests.post(url_with_params, headers=headers, data=payload)

    # Raise error if response failed
    response.raise_for_status()

    return response.json()


def get_timestamp_minutes_ago(minutes):
    """
    Returns a formatted timestamp string for X minutes ago in Eastern timezone.

    Args:
        minutes (int): Number of minutes to subtract from current time

    Returns:
        str: Formatted timestamp string in format 'YYYY-MM-DD HH:MM:SS.000000000'
    """
    eastern = pytz.timezone('US/Eastern')
    current_time_eastern = datetime.datetime.now(eastern)
    time_ago = current_time_eastern - datetime.timedelta(minutes=minutes)
    timestamp_str = time_ago.strftime('%Y-%m-%d %H:%M:%S.000000000')
    return timestamp_str


def replace_placeholders(obj, data_dict):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, (dict, list)):
                replace_placeholders(value, data_dict)
            elif isinstance(value, str) and "{" in value and "}" in value:
                # Extract field name from placeholder {fieldname}
                field_names = re.findall(r'\{([^}]+)\}', value)
                new_value = value

                for field_name in field_names:
                    placeholder = "{" + field_name + "}"
                    if field_name in data_dict and data_dict[field_name] is not None:
                        if field_name == "item_class":
                            # Special handling for item_class
                            class_value = data_dict[field_name]
                            # Convert to int if it's a string representing an integer
                            if isinstance(class_value, str) and class_value.isdigit():
                                class_value = int(class_value)

                            if class_value == 1:
                                mapped_value = "Single Stack CS"
                            elif class_value == 2:
                                mapped_value = "Double Stack CS"
                            else:
                                mapped_value = "Triple Stack CS"

                            new_value = new_value.replace(placeholder, mapped_value)
                        else:
                            # Replace with actual value for other fields
                            new_value = new_value.replace(placeholder, str(data_dict[field_name]))
                    else:
                        # Replace with empty string if field doesn't exist
                        new_value = new_value.replace(placeholder, "")

                obj[key] = new_value
    elif isinstance(obj, list):
        for item in obj:
            replace_placeholders(item, data_dict)

    return obj


def send_netsuite_request(payload, script, deployment):
    secrets = get_secrets()

    # NetSuite credentials
    ns_account_id = secrets["realm"]
    consumer_key = secrets["ConsumerKey"]
    consumer_secret = secrets["ConsumerSecret"]
    token = secrets["ProdTokenID"]
    token_secret = secrets["ProdTokenSecret"]
    baseurl = secrets["baseURL"]

    # NetSuite RESTlet URL - Fixed f-string formatting
    url = f"{baseurl}?script={script}&deploy={deployment}"

    # Create OAuth1 authentication object
    auth = OAuth1(
        client_key=consumer_key,
        client_secret=consumer_secret,
        resource_owner_key=token,
        resource_owner_secret=token_secret,
        signature_method="HMAC-SHA256",
        realm=ns_account_id
    )

    # Set proper headers - match exactly what works in Postman
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    # Convert payload to string to ensure proper formatting
    payload_json = json.dumps(payload)

    # Make POST request
    response = requests.post(
        url,
        headers=headers,
        auth=auth,
        data=payload_json
    )

    # Print response with more details
    print("Status Code:", response.status_code)
    print("Response:", response.text)
    print("Request payload:", payload)
    print("Request headers:", response.request.headers)

    try:
        return response.json()
    except ValueError:
        return {"error": "Invalid JSON response", "text": response.text}


def extract_parts_from_url(url):
    """Extract account name, file share name, and file path from URL"""
    parsed_url = urlparse(url)
    parts = parsed_url.path.strip('/').split('/')

    if len(parts) >= 1:
        account_name = parsed_url.netloc.split('.')[0]
        file_share_name = parts[0]
        file_path = '/'.join(parts[1:]) if len(parts) > 1 else ""

        return {
            'accountName': account_name,
            'fileShareName': file_share_name,
            'filePath': file_path
        }
    return None


def generate_authentication_string(verb, date, account_name, file_share_name, file_path):
    secrets = get_secrets()

    # NetSuite credentials
    ACCOUNT_KEY = secrets["azureFileKey"]

    """Generate the Azure File Storage authentication signature"""
    canonicalized_headers = f"x-ms-date:{date}\nx-ms-version:2019-02-02"
    canonicalized_resource = f"/{account_name}/{file_share_name}/{file_path}"

    string_to_sign = f"{verb}\n\n\n\n\n\n\n\n\n\n\n\n{canonicalized_headers}\n{canonicalized_resource}"

    # Compute HMAC-SHA256 signature
    decoded_key = base64.b64decode(ACCOUNT_KEY)
    signature = base64.b64encode(
        hmac.new(decoded_key, string_to_sign.encode('utf-8'), hashlib.sha256).digest()).decode()

    return f"SharedKey {account_name}:{signature}"


def fetch_azure_file(url):
    """Fetch file from Azure File Storage"""
    parts = extract_parts_from_url(url)
    if not parts:
        raise ValueError("Invalid Azure file URL structure")

    account_name = parts['accountName']
    file_share_name = parts['fileShareName']
    file_path = parts['filePath']

    date = datetime.datetime.utcnow().strftime('%a, %d %b %Y %H:%M:%S GMT')

    auth = generate_authentication_string("GET", date, account_name, file_share_name, file_path)

    headers = {
        "x-ms-date": date,
        "x-ms-version": "2019-02-02",
        "Authorization": auth
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        return response.text
    else:
        raise Exception(f"Failed to fetch file: {response.status_code}, Response: {response.text}")


def parse_shipments_xml(xml_content):
    import xml.etree.ElementTree as ET

    # Parse XML
    root = ET.fromstring(xml_content)

    # Define namespace prefix for easier referencing
    namespaces = {
        'ils': 'http://www.manh.com/ILSNET/Interface'
    }

    # Find all shipment elements
    shipments = root.findall('.//ils:Shipment', namespaces)

    # Create array to hold shipment data
    shipment_data = []

    for shipment in shipments:
        # Skip deleted shipments
        deleted = shipment.find('ils:Deleted', namespaces)
        if deleted is not None and deleted.text == 'Y':
            continue

        # Extract data from each shipment
        shipment_info = {}

        # Date
        creation_date = shipment.find('ils:CreationDateTimeStamp', namespaces)
        if creation_date is not None:
            shipment_info['date'] = creation_date.text

        # InterfaceRecordId
        interface_id = shipment.find('ils:InterfaceRecordId', namespaces)
        if interface_id is not None:
            shipment_info['InterfaceRecordId'] = interface_id.text

        # BolNumAlpha
        bol_num = shipment.find('ils:BolNumAlpha', namespaces)
        if bol_num is not None:
            shipment_info['BolNumAlpha'] = bol_num.text

        # InternalShipmentNum
        ErpOrder = shipment.find('ils:ErpOrder', namespaces)
        if ErpOrder is not None:
            shipment_info['InternalShipmentNum'] = ErpOrder.text

        # ShipToAddress
        ship_to = shipment.find('.//ils:ShipToAddress', namespaces)
        if ship_to is not None:
            address_info = {}
            for address_part in ship_to:
                tag = address_part.tag.split('}')[1]  # Remove namespace
                address_info[tag] = address_part.text
            shipment_info['ShipToAddress'] = address_info

        # Item info
        detail = shipment.find('.//ils:ShipmentDetail', namespaces)
        if detail is not None:
            sku = detail.find('.//ils:SKU', namespaces)
            if sku is not None:
                item = sku.find('ils:Item', namespaces)
                if item is not None:
                    shipment_info['Item'] = item.text

        # Allocation info (use first allocation)
        alloc_requests = shipment.findall('.//ils:ShipmentAllocRequest', namespaces)
        if alloc_requests:
            alloc = alloc_requests[0]
            lot = alloc.find('ils:Lot', namespaces)
            if lot is not None:
                shipment_info['lot'] = lot.text
            alloc_qty = alloc.find('ils:AllocQty', namespaces)
            if alloc_qty is not None:
                shipment_info['AllocQty'] = alloc_qty.text

        # Container info (can be multiple)
        containers = shipment.findall('.//ils:ShippingContainer', namespaces)
        container_list = []
        for container in containers:
            container_info = {}
            container_id = container.find('ils:ContainerId', namespaces)
            if container_id is not None:
                container_info['ContainerId'] = container_id.text

            # Each container might have one or more ContainerDetails
            container_details = container.findall('.//ils:ContainerDetail', namespaces)
            for detail in container_details:
                lot = detail.find('ils:Lot', namespaces)
                qty = detail.find('ils:Quantity', namespaces)
                quantity_um = detail.find('ils:QuantityUm', namespaces)
                if lot is not None:
                    container_info['Lot'] = lot.text
                if qty is not None:
                    container_info['Quantity'] = qty.text
                if quantity_um is not None:
                    container_info['QuantityUM'] = quantity_um.text

            container_list.append(container_info)

        if container_list:
            shipment_info['Containers'] = container_list

        # Add shipment record
        shipment_data.append(shipment_info)

    return shipment_data


def parse_receipts_xml(xml_content):
    # Parse XML
    root = ET.fromstring(xml_content)

    # Define namespace with prefix
    ns = {'manh': 'http://www.manh.com/ILSNET/Interface'}

    # Find all receipt elements
    receipts = root.findall('.//manh:Receipt', ns)
    print(receipts)
    print(f"Found {len(receipts)} receipts in XML.")
    # print("XML Content:", xml_content)

    receipt_data = []

    for receipt in receipts:
        receipt_info = {}

        # Extract top-level fields
        fields = [
            ('CreationDateTimeStamp', 'date'),
            ('InterfaceRecordId', 'InterfaceRecordId'),
            ('ReceiptId', 'ReceiptId'),
            ('InternalRecNum', 'InternalRecNum'),
            ('ReceiptType', 'ReceiptType'),
            ('Warehouse', 'Warehouse')
        ]
        for tag, key in fields:
            el = receipt.find(f'manh:{tag}', ns)
            if el is not None:
                receipt_info[key] = el.text

        # Vendor info
        vendor = receipt.find('manh:Vendor', ns)
        if vendor is not None:
            vendor_info = {}
            ps = vendor.find('manh:ProcessStamp', ns)
            if ps is not None:
                vendor_info['ProcessStamp'] = ps.text
            receipt_info['Vendor'] = vendor_info

        # Details
        detail_lines = []
        for detail in receipt.findall('.//manh:ReceiptDetail', ns):
            detail_info = {}
            line_num = detail.find('manh:InternalRecLineNum', ns)
            qty = detail.find('manh:OriginalQty', ns)
            sku = detail.find('manh:SKU', ns)

            if line_num is not None:
                detail_info['InternalRecLineNum'] = line_num.text
            if qty is not None:
                detail_info['OriginalQty'] = qty.text
            if sku is not None:
                item = sku.find('manh:Item', ns)
                desc = sku.find('manh:Desc', ns)
                if item is not None:
                    detail_info['Item'] = item.text
            detail_lines.append(detail_info)
        if detail_lines:
            receipt_info['DetailLines'] = detail_lines

        # Containers
        container_data = []
        for container in receipt.findall('.//manh:ReceiptContainer', ns):
            container_info = {}
            tags = [
                'ContainerId', 'Lot', 'Qty', 'QtyUm',
                'ExpDate'
            ]
            for tag in tags:
                el = container.find(f'manh:{tag}', ns)
                if el is not None:
                    container_info[tag] = el.text
            container_data.append(container_info)
        if container_data:
            receipt_info['Containers'] = container_data

        receipt_data.append(receipt_info)

    return receipt_data


def generate_control_number():
    """Generate a random 9-digit control number"""
    return str(random.randint(1, 999999999)).zfill(9)


def extract_edi_fields(edi_content):
    """
    Extract necessary fields from the original EDI document to create a 997 response

    Args:
        edi_content (str): The original EDI document content

    Returns:
        dict: Dictionary containing extracted fields
    """
    extracted = {
        "sender_id": "",
        "receiver_id": "",
        "isa_control_number": "",
        "gs_control_number": "",
        "gs_functional_id": "",
        "transaction_set_ids": []
    }

    lines = edi_content.strip().split('\n')

    for line in lines:
        segments = line.split('*')
        if segments[0] == "ISA":
            # For 997, we swap sender and receiver
            extracted["receiver_id"] = segments[6].strip()
            extracted["sender_id"] = segments[8].strip()
            extracted["isa_control_number"] = segments[13].strip()

        elif segments[0] == "GS":
            extracted["gs_functional_id"] = segments[1]
            # For 997, we swap sender and receiver
            extracted["gs_receiver_id"] = segments[2].strip()
            extracted["gs_sender_id"] = segments[3].strip()
            extracted["gs_control_number"] = segments[6]

        elif segments[0] == "ST":
            # Add each transaction set to our list
            if len(segments) > 2:
                extracted["transaction_set_ids"].append((segments[1], segments[2]))

    return extracted


def generate_edi_997(edi_content):
    # Extract relevant fields from the original EDI
    extracted = extract_edi_fields(edi_content)

    # Generate control numbers
    isa_control_number = generate_control_number()
    gs_control_number = generate_control_number()
    transaction_control_number = generate_control_number()

    # Current date and time
    current_date = datetime.datetime.now().strftime("%Y%m%d")
    current_ymd = datetime.datetime.now().strftime("%y%m%d")
    current_time = datetime.datetime.now().strftime("%H%M")

    # Begin building the EDI document
    edi_segments = []

    # Add ISA segment (using our ID as sender, trading partner as receiver)
    # This reverses the original direction
    isa = f"ISA*00*{'':<10}*00*{'':<10}*01*{'119354268':<15}*01*{'081315672':<15}*{current_ymd}*{current_time}*U*00401*{isa_control_number}*0*T*>~"
    edi_segments.append(isa)

    # Add GS segment (using client's ID as sender, our ID as receiver)
    gs = f"GS*FA*119354268*081315672*{current_date}*{current_time}*{gs_control_number}*X*004010~"
    edi_segments.append(gs)

    # ST - Transaction Set Header
    st = f"ST*997*{transaction_control_number}~"
    edi_segments.append(st)

    # AK1 - Functional Group Response Header
    # Use the functional ID from original
    functional_id = extracted.get("gs_functional_id", "PO")
    group_control_number = extracted.get("gs_control_number", "1")
    ak1 = f"AK1*{functional_id}*{group_control_number}~"
    edi_segments.append(ak1)

    # Loop through each original transaction set
    for idx, (transaction_type, transaction_id) in enumerate(
            extracted.get("transaction_set_ids", [("850", "0001")]), 1
    ):
        # AK2 - Transaction Set Response Header
        ak2 = f"AK2*{transaction_type}*{transaction_id}~"
        edi_segments.append(ak2)

        # AK5 - Transaction Set Response Trailer
        # A = Accepted
        ak5 = f"AK5*A~"
        edi_segments.append(ak5)

    # AK9 - Functional Group Response Trailer
    # A = Accepted, count of transaction sets, count received, count accepted
    transaction_count = len(extracted.get("transaction_set_ids", [("850", "0001")]))
    ak9 = f"AK9*A*{transaction_count}*{transaction_count}*{transaction_count}~"
    edi_segments.append(ak9)

    # SE - Transaction Set Trailer
    # Count segments in the transaction set including ST and SE
    segment_count = len(edi_segments) - 2 + 1  # Add 1 for SE itself, exclude ISA and GS
    se = f"SE*{segment_count}*{transaction_control_number}~"
    edi_segments.append(se)

    # Add GE and IEA segments
    ge = f"GE*1*{gs_control_number}~"
    iea = f"IEA*1*{isa_control_number}~"
    edi_segments.append(ge)
    edi_segments.append(iea)

    # Join all segments into a single string
    edi_document = "\n".join(edi_segments)

    return edi_document


def list_sftp_files():
    """
    Connects to the SFTP server and lists all files in the target directory.

    Returns:
        list: List of filenames in the SFTP directory.
    """
    try:
        secrets = get_secrets()
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        ssh_client.connect(
            hostname=secrets["SFTP_HOST"],
            username=secrets["SFTP_USERNAME"],
            password=secrets["SFTP_PASSWORD"],
            port=secrets["SFTP_PORT"]
        )

        sftp_client = ssh_client.open_sftp()
        sftp_client.chdir(secrets["SFTP_PATH"])  # ✅ Change to target directory

        files = sftp_client.listdir()

        sftp_client.close()
        ssh_client.close()

        print("Files on SFTP server:\n")
        for file in files:
            print(file)
        print("\n")

        return files

    except Exception as e:
        print(f"Error connecting to SFTP: {str(e)}")
        return []


def download_sftp_file(sftp_client, filename):
    """
    Downloads a file from the SFTP server.

    Args:
        sftp_client: Active SFTP client.
        filename (str): Name of the file to download.

    Returns:
        str: Content of the downloaded file.
    """
    try:
        file_obj = io.BytesIO()
        sftp_client.getfo(filename, file_obj)
        file_obj.seek(0)
        return file_obj.read().decode("utf-8")
    except Exception as e:
        print(f"Error downloading file {filename}: {str(e)}")
        return None


def move_file_to_archive(sftp_client, filename):
    secrets = get_secrets()
    SFTP_PATH = secrets["SFTP_PATH"]
    SFTP_ARCHIVE_PATH = secrets["SFTP_ARCHIVE_PATH"]

    try:
        source = f"{SFTP_PATH}/{filename}"
        destination = f"{SFTP_ARCHIVE_PATH}/{filename}"
        sftp_client.rename(source, destination)
        print(f"Moved {filename} to archive.")
    except Exception as e:
        print(f"Failed to move {filename} to archive: {str(e)}")


def format_date(date_str):
    """Format date from MM/DD/YYYY to YYYYMMDD"""
    try:
        # Parse the date string
        date_obj = datetime.datetime.strptime(date_str, "%m/%d/%Y")
        # Format to YYYYMMDD
        return date_obj.strftime("%Y%m%d")
    except:
        # Return current date if there's an error
        return datetime.datetime.now().strftime("%Y%m%d")


def send_edi_997(original_edi_content, original_filename=None):
    try:
        secrets = get_secrets()

        # Generate EDI 997 document
        edi_997_content = generate_edi_997(original_edi_content)

        # Create filename based on original and timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Extract transaction type from original filename if possible
        transaction_type = "UNK"
        if original_filename and original_filename.startswith("edi"):
            transaction_type = original_filename[3:6]

        filename = f"997_{transaction_type}_{timestamp}.x12"

        # Connect to SFTP
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        ssh_client.connect(
            hostname=secrets.get("SFTP_HOST"),
            username=secrets.get("SFTP_USERNAME"),
            password=secrets.get("SFTP_PASSWORD"),
            port=secrets.get("SFTP_PORT")
        )

        sftp_client = ssh_client.open_sftp()

        # Convert EDI content to bytes for upload
        edi_bytes = edi_997_content.encode('utf-8')
        file_obj = io.BytesIO(edi_bytes)

        # Build the full remote path to INBOX folder
        remote_path = f"{secrets.get('SFTP_PATH_INBOX')}/{filename}"
        print(f"Uploading 997 acknowledgment to {remote_path}...")

        # Upload the EDI content to the server
        sftp_client.putfo(file_obj, remote_path)

        print(f"Upload complete: {filename}")

        # Close the connections
        sftp_client.close()
        ssh_client.close()

        return True

    except Exception as e:
        print(f"Error generating/sending EDI 997: {str(e)}")
        return False


def group_items_by_tranid(items):
    grouped = defaultdict(list)
    for item in items:
        tranid = item.get('tranid')
        if tranid:
            grouped[tranid].append(item)
    return grouped


def upload_to_sftp(edi_content, filename=None):
    secrets = get_secrets()
    hostname = secrets["SFTP_HOST"]
    username = secrets["SFTP_USERNAME"]
    password = secrets["SFTP_PASSWORD"]
    port = secrets["SFTP_PORT"]
    remote_path_prefix = secrets["SFTP_PATH_INBOX"]

    # Generate a filename with timestamp if none provided
    if not filename:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"945_{timestamp}.x12"

    try:
        # Create an SSH client
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        print(f"Connecting to {hostname}...")
        ssh_client.connect(hostname=hostname, username=username, password=password, port=port)

        # Create an SFTP client
        sftp_client = ssh_client.open_sftp()

        # Convert EDI content to bytes for upload
        edi_bytes = edi_content.encode('utf-8')
        file_obj = io.BytesIO(edi_bytes)

        # Build the full remote path
        full_remote_path = f"{remote_path_prefix}/{filename}"
        print(f"Uploading file to {full_remote_path}...")

        # Upload the EDI content to the server
        sftp_client.putfo(file_obj, full_remote_path)

        print(f"Upload complete: {filename}")

        # Close the connections
        sftp_client.close()
        ssh_client.close()

        return True

    except Exception as e:
        print(f"Error uploading to SFTP: {str(e)}")
        return False


def load_uploaded_ids(log_file):
    """
    Load a set of uploaded transaction IDs from a JSON file.
    If the file doesn't exist, return an empty set.
    """
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            return set(json.load(f))
    return set()


def save_uploaded_id(tranid, log_file):
    """
    Save a transaction ID to the JSON log file, avoiding duplicates.
    """
    ids = load_uploaded_ids(log_file)
    ids.add(tranid)
    with open(log_file, 'w') as f:
        json.dump(sorted(list(ids)), f, indent=2)


def parse_edi_segments(edi_content):
    """
    Cleans and splits EDI string into list of segment dictionaries.
    """
    edi_lines = edi_content.replace('\r\n', '\n').replace('\r', '\n').replace('~', '\n').strip().split('\n')
    segments = []

    for line in edi_lines:
        if not line.strip():
            continue
        parts = line.strip().split("*")
        segment_type = parts[0]
        segment_data = parts[1:]
        segments.append({segment_type: segment_data})

    return segments


def extract_address_fields(segments, target_qualifier="ST"):
    """
    Returns address-related fields for a given N1 qualifier (e.g., ST).
    """
    address = {
        "shipattention": "",
        "shipaddr1": "",
        "shipaddr2": "",
        "shipcity": "",
        "shipstate": "",
        "shipzip": ""
    }

    for i, segment in enumerate(segments):
        if "N1" in segment:
            n1 = segment["N1"]
            if len(n1) >= 2 and n1[0].strip() == target_qualifier:
                address["shipattention"] = n1[1].strip()
                for j in range(i + 1, i + 5):
                    if j >= len(segments):
                        break
                    if "N3" in segments[j]:
                        n3 = segments[j]["N3"]
                        address["shipaddr1"] = n3[0].strip()
                        if len(n3) > 1:
                            address["shipaddr2"] = n3[1].strip()
                    if "N4" in segments[j]:
                        n4 = segments[j]["N4"]
                        address["shipcity"] = n4[0].strip()
                        if len(n4) > 1:
                            address["shipstate"] = n4[1].strip()
                        if len(n4) > 2:
                            address["shipzip"] = n4[2].strip()
                break
    return address


def extract_date_mmddyyyy(date_str):
    """
    Converts a date string from YYYYMMDD to MM/DD/YYYY.
    Returns original string if format doesn't match.
    """
    if len(date_str) == 8 and date_str.isdigit():
        return f"{date_str[4:6]}/{date_str[6:8]}/{date_str[0:4]}"
    return date_str


def was_processed(item_id, filename):
    """
    Check if the item_id was already processed and recorded in the given JSON file.
    """
    try:
        if not os.path.exists(filename):
            return False
        with open(filename, 'r') as f:
            processed = json.load(f)
        return str(item_id) in processed
    except Exception as e:
        print(f"Error checking processed file: {e}")
        return False


def mark_as_processed(item_id, filename):
    """
    Add an item_id to the JSON file to mark it as processed.
    Creates the file if it doesn't exist.
    """
    try:
        processed = []
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                processed = json.load(f)

        if str(item_id) not in processed:
            processed.append(str(item_id))
            with open(filename, 'w') as f:
                json.dump(processed, f, indent=2)
    except Exception as e:
        print(f"Error updating processed file: {e}")


def upload_to_azure_sftp_sojo(edi_content, filename=None):
    import paramiko
    import io
    import datetime

    # SFTP connection details
    hostname = "integrationsuploads.blob.core.windows.net"
    username = "integrationsuploads.sftpuser"
    password = "VHSVIMvkOg0wqjVuomMy0MyyTWTpXlQ1"
    port = 22

    # Filename with timestamp fallback
    if not filename:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"945_{timestamp}.x12"

    try:
        # Set up SSH/SFTP connection
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        print(f"Connecting to {hostname} as {username}...")
        ssh_client.connect(hostname=hostname, username=username, password=password, port=port)
        sftp_client = ssh_client.open_sftp()

        # Convert content to bytes
        edi_bytes = edi_content.encode('utf-8')
        file_obj = io.BytesIO(edi_bytes)

        # Remote path: directly in the user's home (sftp/home/)
        full_remote_path = filename
        print(f"Uploading to {full_remote_path}...")

        sftp_client.putfo(file_obj, full_remote_path)

        print("Upload complete.")

        sftp_client.close()
        ssh_client.close()
        return True

    except Exception as e:
        print(f"Upload failed: {e}")
        return False


def load_or_create_json_file(filename, default_content=None):
    secrets = get_secrets()
    hostname = secrets["sojohostname"]
    username = secrets["sojousername"]
    password = secrets["sojopassword"]
    port = 22

    if default_content is None:
        default_content = {}

    try:
        # Connect to SFTP
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh_client.connect(hostname=hostname, username=username, password=password, port=port)
        sftp_client = ssh_client.open_sftp()

        # Try to read the file
        try:
            with sftp_client.file(filename, mode='r') as f:
                print(f"Loading existing file: {filename}")
                file_content = f.read().decode("utf-8")
                json_data = json.loads(file_content)
        except FileNotFoundError:
            print(f"{filename} not found. Creating with default content...")
            json_bytes = json.dumps(default_content, indent=2).encode("utf-8")
            file_obj = io.BytesIO(json_bytes)
            sftp_client.putfo(file_obj, filename)
            json_data = default_content

        # Cleanup
        sftp_client.close()
        ssh_client.close()
        return json_data

    except Exception as e:
        print(f"Error loading or creating JSON file: {e}")
        return None


def save_json_file(filename, content):
    secrets = get_secrets()
    hostname = secrets["sojohostname"]
    username = secrets["sojousername"]
    password = secrets["sojopassword"]
    port = 22

    try:
        # Establish SSH/SFTP connection
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh_client.connect(hostname=hostname, username=username, password=password, port=port)
        sftp_client = ssh_client.open_sftp()

        # Prepare the JSON content as bytes
        json_bytes = json.dumps(content, indent=2).encode("utf-8")
        file_obj = io.BytesIO(json_bytes)

        # Upload the file
        sftp_client.putfo(file_obj, filename)
        print(f"Successfully saved {filename} to SFTP.")

        # Close connections
        sftp_client.close()
        ssh_client.close()
        return True

    except Exception as e:
        print(f"Failed to save JSON file {filename}: {e}")
        return False


def group_items_by_nstransaction(items):
    grouped = defaultdict(list)
    for item in items:
        nstransaction = item.get('nstransaction')
        if nstransaction:
            grouped[nstransaction].append(item)
    return grouped