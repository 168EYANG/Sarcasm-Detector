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

        // chrome.runtime.sendMessage(
        //     {sentence: tweet.innerHTML}, 
        //     function(response) {
        //         result = response.farewell;
        //     }
        // )

        fetch('url/?input='+tweet.innerHTML)
            .then(response => response.json())
            .then(data => {
                console.log('Data from backend: ', data);
                if (data == "negative") {
                    symbol.textContent = " - ❌ negative";
                } else if (data == "neutral") {
                    symbol.textContent = " - 😐 neutral";
                } else if (data == "positive") {
                    symbol.textContent = " - ✅ positive";
                }
            })
            .catch(error => console.error('Error: ', error));
        
        

        // if (result == [1, 0, 0,]) {
        //     symbol.textContent = " - ❌ negative";
        // } else if (result == [0, 1, 0]) {
        //     symbol.textContent = " - ✅❌ neutral";
        // } else if (result == [0, 0, 1]) {
        //     symbol.textContent = " - ✅ positive";
        // }
        // if (tweet.innerHTML.includes('tenz')) {
        //     symbol.textContent = ' - ❌ /s ';
        // } else {
        //     symbol.textContent = ' - ✅ /srs ';
        // }
        symbol.style.marginRight = '5px';
        // if (!tweet.includes('/srs') && !tweet.includes('/s')) {
        string = tweet.innerHTML + symbol.textContent;
        if (!tweet.innerHTML.includes(' - ❌ negative') && !tweet.innerHTML.includes(' - ✅ positive') && !tweet.innerHTML.includes(' - 😐 neutral')) {
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
