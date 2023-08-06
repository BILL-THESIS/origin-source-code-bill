#!/bin/zsh

text_merge="/Users/billpatcharaprapa/origin-source-code-bill/origin-source-code-bill/Github/jspwiki/merge_base.txt"
text_base="/Users/billpatcharaprapa/origin-source-code-bill/origin-source-code-bill/Github/jspwiki/shiro_base.txt"

host="http://localhost:9000"
token_sonar="sqa_fe106141f0cea2539b49fadd714e32383f29d433"

source="/Users/billpatcharaprapa/jspwiki"

cd $source
	echo "file directory :: $source"

while read line; do
	echo $line

git fetch

git checkout $line

mvn sonar:sonar \
  -Dmaven.test.skip=true \
  -Dsonar.projectKey=jspwiki-$line \
  -Dsonar.projectName=jspwiki-$line \
  -Dsonar.java.binaries=$source \
  -Dsonar.host.url=http://localhost:9000 \
  -Dsonar.token=$token_sonar \
  -Dsonar.login=admin \
  -Dsonar.password=admin21

done < $text_merge