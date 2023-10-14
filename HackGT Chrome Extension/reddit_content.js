const comments = document.querySelectorAll('[data-testid=comment]');

comments.forEach(comment => {
    const symbol = document.createElement('span');
    if (comment.innerHTML.includes('Russia')) {
        symbol.textContent = '/s';
    } else {
        symbol.textContent = '/srs';
    }
    // symbol.textContent = '/s'; // You can change this symbol as needed
    symbol.style.marginRight = '5px'; // Adjust spacing if necessary
    comment.prepend(symbol);
});



