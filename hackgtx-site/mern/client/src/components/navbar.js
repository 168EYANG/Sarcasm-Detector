import React from "react";

import "bootstrap/dist/css/bootstrap.css";

import { NavLink } from "react-router-dom";

export default function Navbar() {
  return (
    <div>
      <nav className="px-4 navbar fixed-top navbar-expand-lg navbar-light bg-light">
        <NavLink className="navbar-brand fw-bold" to="/">
          Customon
        </NavLink>
        <button
          className="navbar-toggler"
          type="button"
          data-toggle="collapse"
          data-target="#navbarSupportedContent"
        >
          <span className="navbar-toggler-icon" />
        </button>

        <div
          className="nav collapse navbar-collapse"
          id="navbarSupportedContent"
        >
          <ul class="navbar-nav">
            <li class="nav-item">
              <NavLink className="nav-link" to="/records">
                View Records
              </NavLink>
            </li>
            <li>
              <NavLink className="nav-link" to="/create">
                Create Record
              </NavLink>
            </li>
          </ul>
        </div>
      </nav>
    </div>
  );
}
