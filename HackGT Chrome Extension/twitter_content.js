// const tweetContent = document.querySelectorAll('[data-testid=tweetText]');
// console.log("YES I EXIST");
// const tweetContent = document.querySelectorAll('div[data-testid=tweetText]');

// console.log(tweetContent[0]);

// setInterval(() => {
//     for (const d of document.querySelectorAll('div[data-testid="like"]')){
//         d.click();
//     }
// }, 5000)

setInterval(() => {
    for (const tweet of document.querySelectorAll('div[data-testid="tweetText"]')) {
        const symbol = document.createElement('span');
        if (tweet.innerHTML.includes('tenz')) {
            symbol.textContent = ' - ❌ /s ';
        } else {
            symbol.textContent = ' - ✅ /srs ';
        }
        symbol.style.marginRight = '5px';
        // if (!tweet.includes('/srs') && !tweet.includes('/s')) {
        string = tweet.innerHTML + symbol.textContent;
        if (!tweet.innerHTML.includes(' - ❌ /s ') && !tweet.innerHTML.includes(' - ✅ /srs ')) {
            tweet.innerHTML = string;
        }
        // prepend(symbol);
        // }

    }
}, 5000)


// tweetContent.forEach(comment => {
//     comment.style.display = "none";
// });

// tweetContent.forEach(comment => {
//     const symbol = document.createElement('span');
//     if (comment.innerHTML.includes('tenz')) {
//         symbol.textContent = '/s';
//     } else {
//         symbol.textContent = '/srs';
//     }
//     // symbol.textContent = '/s'; // You can change this symbol as needed
//     symbol.style.marginRight = '5px'; // Adjust spacing if necessary
//     comment.prepend(symbol);
// });
