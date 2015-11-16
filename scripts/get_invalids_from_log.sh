grep "DEBUG:root:Invalid:" chunking.log | uniq | sed 's/DEBUG:root:Invalid: '//g
