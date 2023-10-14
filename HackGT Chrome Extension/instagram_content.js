const comments = document.querySelectorAll('#a9zs');

comments.forEach(comment => {
    const symbol = document.createElement('span');
    if (comment.innerHTML.includes('That')) {
        symbol.textContent = '/srs';
    } else {
        symbol.textContent = '/s';
    }
    // symbol.textContent = '/s'; // You can change this symbol as needed
    symbol.style.marginRight = '5px'; // Adjust spacing if necessary
    comment.prepend(symbol);
});


