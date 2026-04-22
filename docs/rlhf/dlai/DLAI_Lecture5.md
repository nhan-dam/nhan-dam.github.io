# Google Cloud Setup

> Created on: 22 April 2026
>
> Updated on: 22 April 2026

## 1. Overview

This chapter describes how to configure Google Cloud and Vertex AI for use in machine learning projects. The setup involves creating a project, enabling billing and APIs, provisioning a service account with the appropriate permissions, and initialising the Vertex AI client.

---

## 2. Creating a Google Cloud Project

All usage of Google Cloud services is associated with a project. Projects form the basis for managing APIs, billing, collaborators, and resource permissions.

To create a project, visit the [Cloud Console](https://console.cloud.google.com), where you will be prompted to create one on first login. A [free tier](https://cloud.google.com/free/docs/gcp-free-tier) is available, which includes a 90-day, $300 trial credit.

---

## 3. Enabling Billing

A Cloud Billing account defines who pays for a given set of resources and must be linked to the project before any services can be used. Within your project, navigate to **Billing** in the left-hand menu and follow the prompts to attach a billing account. Detailed instructions are available in the [Google Cloud documentation](https://cloud.google.com/billing/docs/how-to/modify-project).

---

## 4. Enabling APIs

Once billing is active, enable the required APIs by visiting the [API enablement page](https://console.cloud.google.com/flows/enableapi?apiid=aiplatform,iam,bigquery.googleapis.com). The three APIs required for this setup are:

- Vertex AI.
- BigQuery.
- IAM.

---

## 5. Creating a Service Account

A service account is a non-human identity used by applications and compute workloads to authenticate with Google Cloud services. It is identified by a unique email address.

### 5.1. Creating an Account

Navigate to the [Create Service Account](https://console.cloud.google.com/projectselector/iam-admin/serviceaccounts/create) page, select your project, and give the account a name of your choosing.

### 5.2. Granting Permissions

During creation, grant the service account the permissions required for Vertex AI, BigQuery, and IAM access. The exact roles will depend on the scope of your project; refer to the IAM permissions panel in the Cloud Console for guidance.

### 5.3. Creating a JSON Key

After creating the account, generate a key by selecting **ADD KEY → Create new key** and choosing the **JSON** format. The key file will be downloaded immediately. Note that it cannot be downloaded again after this point, so store it securely.

---

## 6. Authenticating with the Credentials File

With the JSON key file in hand, create a credentials object using the `google-auth` library. This object is subsequently passed to the Vertex AI client.

```python
from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials

# Path to the downloaded JSON key file
key_path = 'your_key.json'

credentials = Credentials.from_service_account_file(
    key_path,
    scopes=['https://www.googleapis.com/auth/cloud-platform']
)

if credentials.expired:
    credentials.refresh(Request())
```

---

## 7. Setting Up a Cloud Storage Bucket

Cloud Storage is Google Cloud's object storage service and can hold arbitrary file types (e.g. images, CSVs, saved model artefacts). Data is organised into 'buckets', and every object within Cloud Storage is addressed by a path of the form `gs://{bucket_name}/{path/to/object}`.

To create a bucket:

1. Navigate to [Cloud Storage → Buckets](https://console.cloud.google.com/storage/browser) in the Cloud Console.
2. Click **CREATE**.
3. Provide a unique bucket name and select a region. US or EU multiregion are sensible defaults for most projects.

Once created, the bucket is accessible at `gs://{name_of_your_bucket}`.

---

## 8. Initialising Vertex AI

With credentials and a project ID available, initialise the Vertex AI client as follows. Copy your project ID from the [Cloud Console](https://cloud.google.com/resource-manager/docs/creating-managing-projects) and substitute it below.

```python
import google.cloud.aiplatform as aiplatform

PROJECT_ID = 'your_project_ID'
REGION = 'us-central1'

aiplatform.init(
    project=PROJECT_ID,
    location=REGION,
    credentials=credentials
)
```

After this call, the Vertex AI SDK is authenticated and ready for use within the specified project and region.
