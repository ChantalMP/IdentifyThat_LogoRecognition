import http.client, urllib.request, urllib.parse, urllib.error, base64, json
from PIL import Image
import io
import  requests
from io import BytesIO

# Replace the subscription_key string value with your valid subscription key.
subscription_key = 'caf4d340e76e417f9e5b1fd67f9836fe'

# Replace or verify the region.
#
# You must use the same region in your REST API call as you used to obtain your subscription keys.
# For example, if you obtained your subscription keys from the westus region, replace
# "westcentralus" in the URI below with "westus".
#
# NOTE: Free trial subscription keys are generated in the westcentralus region, so if you are using
# a free trial subscription key, you should not need to change this region.
uri_base = 'https://westcentralus.api.cognitive.microsoft.com/vision/v1.0/'

headers = {
    # Request headers.
    'Content-Type': 'application/octet-stream',
    'Ocp-Apim-Subscription-Key': subscription_key,
}

params = urllib.parse.urlencode({
    # Request parameters. All of them are optional.
    'visualFeatures': 'Tags, Color',
    'language': 'en',
})


img_filename = 'bobo.jpg'
with open(img_filename, 'rb') as f:
    img_data = f.read()


try:
    # Execute the REST API call and get the response.
    api_url = "http://westcentralus.api.cognitive.microsoft.com/vision/v1.0/analyze?%s" % params
    # conn = http.client.HTTPSConnection('westcentralus.api.cognitive.microsoft.com')
    # conn.request("POST", "/vision/v1.0/analyze?%s" % params,img_data, headers)

    response = requests.post(api_url , params=params, headers=headers, data=img_data)
    # response = conn.getresponse()
    # data = response.read()
    print(response.json())
    # 'data' contains the JSON data. The following formats the JSON data for display.
    # parsed = json.loads(response.json())
    # print ("Response:")
    # print (json.dumps(parsed, sort_keys=True, indent=2))
    # conn.close()
    response.close()
except Exception as e:
    print('Error:')
    print(e)

####################################