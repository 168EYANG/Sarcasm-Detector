// const comments = document.querySelectorAll('[data-testid=comment]');

setInterval(() => {
    for (const comment of document.querySelectorAll('div[data-testid="comment"]')) {
        const symbol = document.createElement('span');
        if (comment.innerHTML.includes('tenz')) {
            symbol.textContent = ' - ❌ /s ';
        } else {
            symbol.textContent = ' - ✅ /srs ';
        }
        symbol.style.marginRight = '5px';
        // if (!tweet.includes('/srs') && !tweet.includes('/s')) {
        string = comment.innerHTML + symbol.textContent;
        if (!comment.innerHTML.includes(' - ❌ /s ') && !comment.innerHTML.includes(' - ✅ /srs ')) {
            comment.innerHTML = string;
        }
        // prepend(symbol);
        // }

    }
}, 5000)

