setInterval(() => {
    for (const comment of document.querySelectorAll('div[id="comment-content"]')) {
        const symbol = document.createElement('span');
        if (comment.textContent.includes('tenz')) {
            symbol.textContent = ' - ❌ /s ';
        } else {
            symbol.textContent = ' - ✅ /srs ';
        }
        symbol.style.marginRight = '5px';
        // if (!tweet.includes('/srs') && !tweet.includes('/s')) {
        console.log("Comment is " + comment.textContent);


        string = symbol.textContent + comment.textContent;
        if (!comment.textContent.includes(' - ❌ /s ') && !comment.textContent.includes(' - ✅ /srs ')) {
            comment.textContent = (comment.textContent + symbol.textContent);
            comment.textContent = comment.textContent.replace("Show less", " ");
            comment.textContent = comment.textContent.replace("Read more", " ");
            comment.style.fontSize = "10pt";
            comment.style.color = "#fff";
        }
        // prepend(symbol);
        // }

    }
}, 5000)
