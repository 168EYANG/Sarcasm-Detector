// const comments = document.querySelectorAll('[data-testid=comment]');

setInterval(() => {
    for (const comment of document.querySelectorAll('div[data-testid="comment"]')) {
        const symbol = document.createElement('span');

        fetch('https://en.wikipedia.org/wiki/Operation_Barbarossa')
        .then(response => response.json())
        .then(data => {
            console.log('Data from backend: ', data);
            // if (data == "negative") {
            //     symbol.textContent = " - ❌ negative";
            // } else if (data == "neutral") {
            //     symbol.textContent = " - 😐 neutral";
            // } else if (data == "positive") {
            //     symbol.textContent = " - ✅ positive";
            // }
        })
        .catch(error => console.error('Error: ', error));

        if (comment.innerHTML.includes('tenz')) {
            symbol.textContent = ' - ❌ /s ';
        } else {
            symbol.textContent = ' - ✅ /srs ';
        }
        symbol.style.marginRight = '5px';
        // if (!tweet.includes('/srs') && !tweet.includes('/s')) {
        string = comment.textContent + symbol.textContent;
        if (!comment.innerHTML.includes(' - ❌ /s ') && !comment.innerHTML.includes(' - ✅ /srs ')) {
            comment.innerHTML = string;
        }
        // prepend(symbol);
        // }

    }
}, 5000)

