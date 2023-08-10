#!/bin/sh

text_base="dubbo_base.txt"
text_merge="dubbo_merge.text"

hostURL="http://localhost:9000"
token="squ_6061f03ee4fd6597b16f9c49f6973af71525a796"
sourceDir="/home/bill/dubbo"

token_line="dsIcr3W7g1oMFH5XurbULg2AWfE9xsLAAjchWfFrxnm"

cd $sourceDir
echo "Source file is ::: $sourceDir"

while read line; do
        echo "line no. : $line"
line_sha=${line::-1}

git fetch
git status

git checkout $line_sha
curl -X POST -H "Authorization: Bearer $token_line" -F "message= Checkout  : $line_sha" https://notify-api.line.me/api/notify
git config advice.objectNameWarning false

mvn sonar:sonar \
  -Dsonar.projectKey=dubbo-$line_sha \
  -Dsonar.projectName=dubbo-$line_sha \
  -Dsonar.java.binaries=$sourceDir \
  -Dsonar.host.url=$hostURL \
  -Dsonar.token=$token \
  -Dsonar.projectVersion=$line_sha \
  -Dsonar.scm.revision=$line_sha \
  -Dsonar.scm.disabled=true \
  -Dsonar.login=admin \
  -Dsonar.password=admin21 \

curl -X POST -H "Authorization: Bearer $token_line" -F "message= Sha Parants Begin Build success ozone : $line_sha" https://notify-api.line.me/api/notify
done < $text_base
