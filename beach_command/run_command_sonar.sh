#!/bin/zsh

text_merge="/Users/bill/origin-source-code-bill/models/output/tracking_api_to_sonar/pulsar_nan_column_end.txt"
text_base="/Users/bill/origin-source-code-bill/models/output/tracking_api_to_sonar/pulsar_nan_column_start.txt"
host="https://eserg-sonarqube.dto.technology"
token_sonar="sqa_80671462b7276a63d28d08b1fcca5d7594adb340"
token_line="sTphJfgqNg0XgkbhA2iy9olSzW0fWXvyx7pswSywab1"

source="/Users/bill/pulsar"

cd $source
	echo "file directory :: $source"

while read line; do
        echo "line no. : $line"

git fetch
git status

git checkout $line
  echo "checkout : $line"

git status
curl -X POST -H "Authorization: Bearer $token_line" -F "message= Checkout: $line" https://notify-api.line.me/api/notify
#zgit config advice.objectNameWarning false

mvn sonar:sonar \
  -Dmaven.test.skip=true \
  -Dsonar.projectKey=pulsar-$line \
  -Dsonar.projectName=pulsar-$line \
  -Dsonar.java.binaries=$source \
  -Dsonar.host.url=$host \
  -Dsonar.token=$token_sonar \
  -Dsonar.login=admin \
  -Dsonar.password=mgphev123

curl -X POST -H "Authorization: Bearer $token_line" -F "message= Sha Build success pulsar : $line" https://notify-api.line.me/api/notify
done < $text_base