#!/bin/zsh

text="/Users/bill/origin-source-code-bill/beach_command/output/ozone_failed.text"
host="https://eserg-sonarqube.dto.technology"
token_sonar="sqa_80671462b7276a63d28d08b1fcca5d7594adb340"
token_line="sTphJfgqNg0XgkbhA2iy9olSzW0fWXvyx7pswSywab1"

source="/Users/bill/ozone"
log_dir="/Users/bill/origin-source-code-bill/beach_command/output/logs_ozone"


mkdir -p $log_dir

cd $source
echo "file directory :: $source"



while read line; do
    echo "Processing line no. : $line"

    # Fetch and log git status
    git fetch
    git status

    # Checkout and log
    git checkout $line
    echo "Checkout to : $line"

    # Log git status after checkout
    git status

    # Notify via LINE
    curl -X POST -H "Authorization: Bearer $token_line" -F "message=Checkout: $line" https://notify-api.line.me/api/notify


    mvn -U dependency:copy -Dartifact=com.foo:my-foo:LATEST


    # Run sonar scan and log
    mvn sonar:sonar \
        -Dmaven.test.skip=true \
        -Dsonar.projectKey=ozone-$line \
        -Dsonar.projectName=ozone-$line \
        -Dsonar.java.binaries=$source \
        -Dsonar.host.url=$host \
        -Dsonar.token=$token_sonar \
        -Dsonar.login=admin \
        -Dsonar.password=mgphev123 &> "$log_dir/sonar_scan_failed$line.log"

    # Check for build failure in the log
    if grep -q "BUILD FAILURE" "$log_dir/sonar_scan_$line.log"; then
        echo "Sonar build failed for $line" >> "$log_dir/sonar_failed.log"
        curl -X POST -H "Authorization: Bearer $token_line" -F "message=Sonar build failed for $line" https://notify-api.line.me/api/notify
    fi

    sleep 5

done < $text
