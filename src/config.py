import os
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods

# IBM Watsonx AI Credentials
CREDENTIALS = {
    "url": os.getenv("IBM_CLOUD_URL", "https://us-south.ml.cloud.ibm.com"),
    "api_key": os.getenv("IBM_CLOUD_API_KEY", "")
}

PROJECT_ID = os.getenv("WATSONX_PROJECT_ID", "skills-network")

# Model Configuration
DEFAULT_MODEL_ID = 'ibm/granite-3-3-8b-instruct'

DEFAULT_PARAMETERS = {
    GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
    GenParams.MIN_NEW_TOKENS: 1,
    GenParams.MAX_NEW_TOKENS: 256,
    GenParams.TEMPERATURE: 0.5
}
